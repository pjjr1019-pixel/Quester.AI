"""Bounded local audio helpers for optional voice-input workflows."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

from data_structures import (
    AudioSynthesisResult,
    AudioTranscriptionResult,
    VoiceActivityReport,
    VoiceActivitySegment,
    utc_now,
)


def system_speech_available() -> bool:
    """Return whether the Windows System.Speech path is plausibly available."""
    return os.name == "nt" and shutil.which("powershell") is not None


def analyze_voice_activity(
    audio_path: Path | str,
    *,
    max_duration_s: float,
    frame_ms: int,
    min_speech_ms: int,
    merge_silence_ms: int,
    energy_threshold: float,
    analyzer_backend: str,
) -> VoiceActivityReport:
    """Return a deterministic amplitude-based VAD report for one WAV file."""
    source_path = Path(audio_path)
    try:
        samples, sample_rate_hz, channel_count, duration_seconds, analyzed_duration_seconds = _read_wav_mono(
            source_path,
            max_duration_s=max_duration_s,
        )
    except Exception as exc:
        return VoiceActivityReport(
            source_path=str(source_path),
            analyzer_backend=analyzer_backend,
            warnings=(str(exc),),
            updated_at=utc_now(),
        )

    if not samples:
        return VoiceActivityReport(
            source_path=str(source_path),
            analyzer_backend=analyzer_backend,
            sample_rate_hz=sample_rate_hz,
            channel_count=channel_count,
            duration_seconds=duration_seconds,
            analyzed_duration_seconds=analyzed_duration_seconds,
            warnings=("no_audio_samples_loaded",),
            updated_at=utc_now(),
        )

    frame_size = max(1, int(sample_rate_hz * (frame_ms / 1000.0)))
    frame_energies = [
        sum(abs(sample) for sample in samples[offset : offset + frame_size]) / max(
            1,
            len(samples[offset : offset + frame_size]),
        )
        for offset in range(0, len(samples), frame_size)
    ]
    if not frame_energies:
        return VoiceActivityReport(
            source_path=str(source_path),
            analyzer_backend=analyzer_backend,
            sample_rate_hz=sample_rate_hz,
            channel_count=channel_count,
            duration_seconds=duration_seconds,
            analyzed_duration_seconds=analyzed_duration_seconds,
            warnings=("no_audio_frames_loaded",),
            updated_at=utc_now(),
        )

    sorted_energies = sorted(frame_energies)
    quiet_frame_count = max(1, len(sorted_energies) // 20)
    noise_floor = (
        sum(sorted_energies[:quiet_frame_count]) / quiet_frame_count
        if sorted_energies
        else 0.0
    )
    peak_energy = sorted_energies[-1]
    dynamic_threshold = max(
        energy_threshold,
        noise_floor * 3.0,
        peak_energy * 0.08 if peak_energy > 0.0 else 0.0,
    )
    if peak_energy > 0.0:
        dynamic_threshold = min(dynamic_threshold, peak_energy * 0.75)
    speech_flags = [energy >= dynamic_threshold for energy in frame_energies]
    merge_gap_frames = max(0, math.ceil(merge_silence_ms / max(1, frame_ms)))
    min_speech_frames = max(1, math.ceil(min_speech_ms / max(1, frame_ms)))

    raw_segments = _collect_speech_frame_segments(speech_flags)
    merged_segments = _merge_frame_segments(raw_segments, max_gap_frames=merge_gap_frames)
    segments: list[VoiceActivitySegment] = []
    speech_frames = 0
    for start_frame, end_frame in merged_segments:
        frame_count = (end_frame - start_frame) + 1
        if frame_count < min_speech_frames:
            continue
        speech_frames += frame_count
        segments.append(
            VoiceActivitySegment(
                start_ms=int(start_frame * frame_ms),
                end_ms=int(min((end_frame + 1) * frame_ms, analyzed_duration_seconds * 1000.0)),
                mean_abs_level=round(
                    sum(frame_energies[start_frame : end_frame + 1]) / max(1, frame_count),
                    4,
                ),
            )
        )

    warnings: list[str] = []
    if duration_seconds > analyzed_duration_seconds:
        warnings.append("audio_clipped_to_duration_limit")
    if not segments:
        warnings.append("no_speech_detected")

    return VoiceActivityReport(
        source_path=str(source_path),
        analyzer_backend=analyzer_backend,
        sample_rate_hz=sample_rate_hz,
        channel_count=channel_count,
        duration_seconds=duration_seconds,
        analyzed_duration_seconds=analyzed_duration_seconds,
        speech_ratio=round(speech_frames / max(1, len(frame_energies)), 4),
        segment_count=len(segments),
        segments=tuple(segments),
        warnings=tuple(warnings),
        updated_at=utc_now(),
    )


def transcribe_with_stub(
    audio_path: Path | str,
    *,
    transcription_model: str,
    vad_report: VoiceActivityReport,
    used_vad: bool,
    max_transcript_chars: int,
) -> AudioTranscriptionResult:
    """Return a deterministic local transcript derived from the audio file name."""
    source_path = Path(audio_path)
    warnings = list(vad_report.warnings)
    transcript_text = _stub_transcript_from_path(source_path)
    status = "transcribed"
    if used_vad and vad_report.segment_count == 0:
        transcript_text = ""
        status = "no_speech"
    elif not transcript_text:
        transcript_text = "audio clip with detected speech"
        warnings.append("stub_transcript_generated_from_generic_audio_label")
    transcript_text = transcript_text[:max_transcript_chars].strip()
    return AudioTranscriptionResult(
        source_path=str(source_path),
        status=status,
        transcript_text=transcript_text,
        normalized_question=normalize_transcript_to_question(transcript_text),
        transcription_backend="stub_speech_to_text",
        transcription_model=transcription_model,
        used_vad=used_vad,
        duration_seconds=vad_report.duration_seconds,
        analyzed_duration_seconds=vad_report.analyzed_duration_seconds,
        voice_activity=vad_report,
        warnings=tuple(warnings),
        updated_at=utc_now(),
    )


def transcribe_with_system_speech(
    audio_path: Path | str,
    *,
    transcription_model: str,
    vad_report: VoiceActivityReport,
    used_vad: bool,
    max_duration_s: float,
    timeout_s: float,
    max_transcript_chars: int,
) -> AudioTranscriptionResult:
    """Transcribe a bounded WAV file through the local Windows System.Speech recognizer."""
    source_path = Path(audio_path)
    warnings = list(vad_report.warnings)
    if used_vad and vad_report.segment_count == 0:
        return AudioTranscriptionResult(
            source_path=str(source_path),
            status="no_speech",
            transcription_backend="windows_system_speech",
            transcription_model=transcription_model,
            used_vad=True,
            duration_seconds=vad_report.duration_seconds,
            analyzed_duration_seconds=vad_report.analyzed_duration_seconds,
            voice_activity=vad_report,
            warnings=tuple(warnings),
            updated_at=utc_now(),
        )

    if used_vad and vad_report.segments:
        clip_start_s = vad_report.segments[0].start_ms / 1000.0
        clip_end_s = vad_report.segments[-1].end_ms / 1000.0
        clip_end_s = min(clip_end_s, max_duration_s, vad_report.analyzed_duration_seconds)
        clip_start_s = max(0.0, min(clip_start_s, clip_end_s))
    else:
        clip_start_s = 0.0
        clip_end_s = min(max_duration_s, vad_report.analyzed_duration_seconds or max_duration_s)
    clip_path, cleanup_path = _prepare_transcription_clip(
        source_path,
        start_s=clip_start_s,
        end_s=clip_end_s,
        max_duration_s=max_duration_s,
    )
    try:
        transcript_text, backend_warnings = _run_system_speech_transcription(
            clip_path,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        warnings.append(str(exc))
        return AudioTranscriptionResult(
            source_path=str(source_path),
            status="failed",
            transcription_backend="windows_system_speech",
            transcription_model=transcription_model,
            used_vad=True,
            duration_seconds=vad_report.duration_seconds,
            analyzed_duration_seconds=vad_report.analyzed_duration_seconds,
            voice_activity=vad_report,
            warnings=tuple(warnings),
            updated_at=utc_now(),
        )
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)

    warnings.extend(backend_warnings)
    transcript_text = transcript_text[:max_transcript_chars].strip()
    status = "transcribed" if transcript_text else "failed"
    if not transcript_text:
        warnings.append("system_speech_returned_empty_transcript")
    return AudioTranscriptionResult(
        source_path=str(source_path),
        status=status,
        transcript_text=transcript_text,
        normalized_question=normalize_transcript_to_question(transcript_text),
        transcription_backend="windows_system_speech",
        transcription_model=transcription_model,
        used_vad=used_vad,
        duration_seconds=vad_report.duration_seconds,
        analyzed_duration_seconds=vad_report.analyzed_duration_seconds,
        voice_activity=vad_report,
        warnings=tuple(warnings),
        updated_at=utc_now(),
    )


def synthesize_with_stub(
    text: str,
    *,
    output_path: Path | str,
    synthesis_model: str,
    max_chars: int,
    sample_rate_hz: int,
) -> AudioSynthesisResult:
    """Write a deterministic local WAV placeholder for one bounded TTS request."""
    target_path = _prepare_tts_output_path(Path(output_path))
    clipped_text = " ".join(str(text).strip().split())[:max_chars].strip()
    if not clipped_text:
        return AudioSynthesisResult(
            target_path=str(target_path),
            status="failed",
            source_text=str(text),
            clipped_text="",
            synthesis_backend="stub_text_to_speech",
            synthesis_model=synthesis_model,
            warnings=("no_text_to_speak",),
            updated_at=utc_now(),
        )
    warnings: list[str] = []
    if clipped_text != " ".join(str(text).strip().split()):
        warnings.append("tts_text_clipped_to_char_limit")
    duration_seconds = _write_stub_speech_wav(
        clipped_text,
        target_path=target_path,
        sample_rate_hz=sample_rate_hz,
    )
    return AudioSynthesisResult(
        target_path=str(target_path),
        status="synthesized",
        source_text=str(text),
        clipped_text=clipped_text,
        synthesis_backend="stub_text_to_speech",
        synthesis_model=synthesis_model,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        warnings=tuple(warnings),
        updated_at=utc_now(),
    )


def synthesize_with_system_speech(
    text: str,
    *,
    output_path: Path | str,
    synthesis_model: str,
    max_chars: int,
    timeout_s: float,
) -> AudioSynthesisResult:
    """Synthesize one bounded text clip to a local WAV file with Windows System.Speech."""
    target_path = _prepare_tts_output_path(Path(output_path))
    clipped_text = " ".join(str(text).strip().split())[:max_chars].strip()
    if not clipped_text:
        return AudioSynthesisResult(
            target_path=str(target_path),
            status="failed",
            source_text=str(text),
            clipped_text="",
            synthesis_backend="windows_system_speech",
            synthesis_model=synthesis_model,
            warnings=("no_text_to_speak",),
            updated_at=utc_now(),
        )
    warnings: list[str] = []
    if clipped_text != " ".join(str(text).strip().split()):
        warnings.append("tts_text_clipped_to_char_limit")
    try:
        _run_system_speech_synthesis(
            clipped_text,
            output_path=target_path,
            timeout_s=timeout_s,
        )
        duration_seconds, sample_rate_hz = _read_synthesized_wav_metadata(target_path)
    except Exception as exc:
        warnings.append(str(exc))
        return AudioSynthesisResult(
            target_path=str(target_path),
            status="failed",
            source_text=str(text),
            clipped_text=clipped_text,
            synthesis_backend="windows_system_speech",
            synthesis_model=synthesis_model,
            warnings=tuple(warnings),
            updated_at=utc_now(),
        )
    return AudioSynthesisResult(
        target_path=str(target_path),
        status="synthesized",
        source_text=str(text),
        clipped_text=clipped_text,
        synthesis_backend="windows_system_speech",
        synthesis_model=synthesis_model,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        warnings=tuple(warnings),
        updated_at=utc_now(),
    )


def normalize_transcript_to_question(transcript_text: str) -> str:
    """Return a compact question-form string for transcript-to-task import."""
    normalized = " ".join(transcript_text.strip().split())
    if not normalized:
        return ""
    if normalized[-1] in ".!?":
        return normalized
    return f"{normalized}?"


def _read_wav_mono(
    source_path: Path,
    *,
    max_duration_s: float,
) -> tuple[list[float], int, int, float, float]:
    if source_path.suffix.lower() != ".wav":
        raise ValueError("Only local .wav audio files are supported in the current voice-input path.")
    with wave.open(str(source_path), "rb") as wav_file:
        sample_rate_hz = wav_file.getframerate()
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        duration_seconds = frame_count / float(sample_rate_hz or 1)
        max_frames = min(frame_count, int(max_duration_s * sample_rate_hz))
        analyzed_duration_seconds = max_frames / float(sample_rate_hz or 1)
        raw_frames = wav_file.readframes(max_frames)

    samples = _decode_pcm_samples(
        raw_frames,
        sample_width=sample_width,
        channel_count=channel_count,
    )
    return samples, sample_rate_hz, channel_count, duration_seconds, analyzed_duration_seconds


def _decode_pcm_samples(
    raw_frames: bytes,
    *,
    sample_width: int,
    channel_count: int,
) -> list[float]:
    if sample_width == 1:
        values = [((value - 128) / 128.0) for value in raw_frames]
    elif sample_width == 2:
        count = len(raw_frames) // 2
        values = [value / 32768.0 for value in struct.unpack(f"<{count}h", raw_frames)]
    elif sample_width == 4:
        count = len(raw_frames) // 4
        values = [value / 2147483648.0 for value in struct.unpack(f"<{count}i", raw_frames)]
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    if channel_count <= 1:
        return values

    mono_samples: list[float] = []
    for offset in range(0, len(values), channel_count):
        frame_values = values[offset : offset + channel_count]
        if not frame_values:
            continue
        mono_samples.append(sum(frame_values) / len(frame_values))
    return mono_samples


def _collect_speech_frame_segments(flags: list[bool]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    current_start: int | None = None
    for index, is_speech in enumerate(flags):
        if is_speech:
            if current_start is None:
                current_start = index
            continue
        if current_start is not None:
            segments.append((current_start, index - 1))
            current_start = None
    if current_start is not None:
        segments.append((current_start, len(flags) - 1))
    return segments


def _merge_frame_segments(
    segments: list[tuple[int, int]],
    *,
    max_gap_frames: int,
) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged: list[tuple[int, int]] = [segments[0]]
    for start_frame, end_frame in segments[1:]:
        previous_start, previous_end = merged[-1]
        if start_frame - previous_end - 1 <= max_gap_frames:
            merged[-1] = (previous_start, end_frame)
        else:
            merged.append((start_frame, end_frame))
    return merged


def _prepare_transcription_clip(
    source_path: Path,
    *,
    start_s: float,
    end_s: float,
    max_duration_s: float,
) -> tuple[Path, Path | None]:
    clip_duration_s = max(0.0, min(end_s, start_s + max_duration_s) - start_s)
    if start_s <= 0.0 and end_s <= max_duration_s:
        return source_path, None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()
    temp_path = Path(temp_file.name)
    with wave.open(str(source_path), "rb") as source_wav, wave.open(str(temp_path), "wb") as target_wav:
        frame_rate = source_wav.getframerate()
        start_frame = int(start_s * frame_rate)
        frame_count = int(max(0.0, clip_duration_s) * frame_rate)
        source_wav.setpos(min(start_frame, source_wav.getnframes()))
        target_wav.setnchannels(source_wav.getnchannels())
        target_wav.setsampwidth(source_wav.getsampwidth())
        target_wav.setframerate(frame_rate)
        target_wav.writeframes(source_wav.readframes(frame_count))
    return temp_path, temp_path


def _run_system_speech_transcription(
    audio_path: Path,
    *,
    timeout_s: float,
) -> tuple[str, tuple[str, ...]]:
    if not system_speech_available():
        raise RuntimeError("windows_system_speech_unavailable")
    script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$grammar = New-Object System.Speech.Recognition.DictationGrammar
$recognizer.LoadGrammar($grammar)
$recognizer.SetInputToWaveFile($env:QUESTER_AUDIO_PATH)
$segments = @()
while ($true) {
    $result = $recognizer.Recognize()
    if ($null -eq $result) { break }
    if ($result.Text) { $segments += $result.Text }
}
[pscustomobject]@{
    transcript = ($segments -join ' ').Trim()
    segment_count = $segments.Count
} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env={
            **os.environ,
            **dict(QUESTER_AUDIO_PATH=str(audio_path)),
        },
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"windows_system_speech_failed:{stderr or 'unknown_error'}")
    payload = json.loads(completed.stdout.strip() or "{}")
    transcript = str(payload.get("transcript", "")).strip()
    warnings: list[str] = []
    if int(payload.get("segment_count", 0) or 0) <= 0:
        warnings.append("system_speech_returned_no_segments")
    return transcript, tuple(warnings)


def _stub_transcript_from_path(source_path: Path) -> str:
    stem = source_path.stem
    normalized = re.sub(r"[_\-]+", " ", stem)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\b(audio|voice|clip|recording|sample)\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" -_")
    return normalized


def _prepare_tts_output_path(target_path: Path) -> Path:
    resolved = Path(target_path)
    if resolved.suffix.lower() != ".wav":
        raise ValueError("Only local .wav output files are supported in the current text-to-speech path.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_stub_speech_wav(
    text: str,
    *,
    target_path: Path,
    sample_rate_hz: int,
) -> float:
    words = [item for item in re.split(r"\s+", text) if item]
    if not words:
        words = ["speech"]
    frames: list[int] = []
    for index, word in enumerate(words):
        frequency_hz = 220.0 + (index % 5) * 45.0
        duration_s = min(0.28, 0.08 + (len(word) * 0.015))
        tone_frame_count = max(1, int(sample_rate_hz * duration_s))
        for frame_index in range(tone_frame_count):
            envelope = min(1.0, frame_index / max(1, tone_frame_count // 8))
            tail = min(1.0, (tone_frame_count - frame_index) / max(1, tone_frame_count // 8))
            amplitude = 0.45 * min(envelope, tail)
            sample = int(
                32767
                * amplitude
                * math.sin(2.0 * math.pi * frequency_hz * (frame_index / float(sample_rate_hz)))
            )
            frames.append(sample)
        frames.extend([0] * max(1, int(sample_rate_hz * 0.03)))
    with wave.open(str(target_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(struct.pack(f"<{len(frames)}h", *frames))
    return len(frames) / float(sample_rate_hz or 1)


def _run_system_speech_synthesis(
    text: str,
    *,
    output_path: Path,
    timeout_s: float,
) -> None:
    if not system_speech_available():
        raise RuntimeError("windows_system_speech_unavailable")
    script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$outputPath = $env:QUESTER_TTS_OUTPUT_PATH
$text = $env:QUESTER_TTS_TEXT
$synth.SetOutputToWaveFile($outputPath)
$synth.Speak($text)
$synth.Dispose()
[pscustomobject]@{
    target_path = $outputPath
    wrote_file = (Test-Path $outputPath)
} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env={
            **os.environ,
            **dict(
                QUESTER_TTS_OUTPUT_PATH=str(output_path),
                QUESTER_TTS_TEXT=text,
            ),
        },
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"windows_system_speech_tts_failed:{stderr or 'unknown_error'}")
    payload = json.loads(completed.stdout.strip() or "{}")
    if not bool(payload.get("wrote_file", False)) or not output_path.exists():
        raise RuntimeError("windows_system_speech_tts_failed:no_output_file")


def _read_synthesized_wav_metadata(target_path: Path) -> tuple[float, int]:
    with wave.open(str(target_path), "rb") as wav_file:
        sample_rate_hz = wav_file.getframerate()
        frame_count = wav_file.getnframes()
    return frame_count / float(sample_rate_hz or 1), sample_rate_hz
