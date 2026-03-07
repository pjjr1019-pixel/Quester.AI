"""Bounded web-search adapters used by the researcher's optional fallback path."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from config import APP_CONFIG, AppConfig
from retrieval import stable_hash
from runtime_errors import WebLookupError, WebLookupTimeoutError


def _strip_html(value: str) -> str:
    """Remove basic markup from API snippets and collapse whitespace."""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(unescape(without_tags).split())


@dataclass(slots=True, frozen=True)
class WebDocument:
    """Single normalized web result returned by the adapter."""

    title: str
    url: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class WebSearchResponse:
    """Bounded search response with degraded-mode metadata."""

    provider: str
    query: str
    results: tuple[WebDocument, ...]
    degraded: bool = False
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class WebSearchAdapter(Protocol):
    """Behavior required by optional web lookup adapters."""

    provider_name: str

    async def search(self, query: str, *, max_results: int) -> WebSearchResponse: ...


class StubWebSearchAdapter:
    """Deterministic no-network web adapter for stub mode and local tests."""

    provider_name = "stub_web"

    def __init__(self, config: AppConfig = APP_CONFIG) -> None:
        self.config = config

    async def search(self, query: str, *, max_results: int) -> WebSearchResponse:
        result_limit = max(1, min(max_results, self.config.web.max_results_per_query))
        results: list[WebDocument] = []
        for index in range(result_limit):
            url = f"https://stub.example/web/{index + 1}"
            results.append(
                WebDocument(
                    title=f"Stub web result {index + 1}",
                    url=url,
                    content=(
                        f"Stub web evidence #{index + 1} for question: {query}. "
                        "This deterministic fallback exists so stub-mode tests stay portable."
                    ),
                    score=round(max(0.3, 0.82 - (index * 0.06)), 2),
                    metadata={
                        "provider": self.provider_name,
                        "rank": index + 1,
                        "query": query,
                    },
                )
            )
        return WebSearchResponse(
            provider=self.provider_name,
            query=query,
            results=tuple(results),
            metadata={"attempt_count": 1, "result_limit": result_limit},
        )


class WikipediaWebSearchAdapter:
    """Bounded web adapter backed by the public MediaWiki search API."""

    provider_name = "wikipedia"

    def __init__(self, config: AppConfig = APP_CONFIG) -> None:
        self.config = config

    async def search(self, query: str, *, max_results: int) -> WebSearchResponse:
        result_limit = max(1, min(max_results, self.config.web.max_results_per_query))
        try:
            return await asyncio.to_thread(self._search_blocking, query, result_limit)
        except Exception as exc:  # pragma: no cover - defensive fallback
            return WebSearchResponse(
                provider=self.provider_name,
                query=query,
                results=(),
                degraded=True,
                warnings=(f"web lookup failed unexpectedly: {type(exc).__name__}",),
                metadata={"attempt_count": 1, "result_limit": result_limit},
            )

    def _search_blocking(self, query: str, result_limit: int) -> WebSearchResponse:
        warnings: list[str] = []
        attempt_count = 0
        try:
            search_payload = self._request_json(
                {
                    "action": "query",
                    "list": "search",
                    "format": "json",
                    "utf8": "1",
                    "srlimit": str(result_limit),
                    "srprop": "snippet",
                    "srsearch": query,
                }
            )
            attempt_count += 1
        except WebLookupError as exc:
            warnings.append(str(exc))
            return WebSearchResponse(
                provider=self.provider_name,
                query=query,
                results=(),
                degraded=True,
                warnings=tuple(warnings),
                metadata={"attempt_count": max(1, attempt_count), "result_limit": result_limit},
            )

        search_hits = search_payload.get("query", {}).get("search", [])
        if not isinstance(search_hits, list) or not search_hits:
            return WebSearchResponse(
                provider=self.provider_name,
                query=query,
                results=(),
                metadata={"attempt_count": max(1, attempt_count), "result_limit": result_limit},
            )

        page_ids = [str(hit.get("pageid")) for hit in search_hits if hit.get("pageid") is not None]
        extracts_by_id: dict[str, dict[str, Any]] = {}
        if page_ids:
            try:
                extract_payload = self._request_json(
                    {
                        "action": "query",
                        "prop": "extracts|info",
                        "format": "json",
                        "utf8": "1",
                        "inprop": "url",
                        "explaintext": "1",
                        "exintro": "1",
                        "exchars": str(self.config.web.max_extract_chars),
                        "pageids": "|".join(page_ids),
                    }
                )
                attempt_count += 1
                raw_pages = extract_payload.get("query", {}).get("pages", {})
                if isinstance(raw_pages, dict):
                    extracts_by_id = {
                        str(page_id): page
                        for page_id, page in raw_pages.items()
                        if isinstance(page, dict)
                    }
            except WebLookupError as exc:
                warnings.append(str(exc))

        dedupe_keys: set[str] = set()
        results: list[WebDocument] = []
        for rank, hit in enumerate(search_hits, start=1):
            page_id = str(hit.get("pageid", ""))
            title = " ".join(str(hit.get("title", "")).split())
            snippet = _strip_html(str(hit.get("snippet", "")))
            page = extracts_by_id.get(page_id, {})
            full_url = str(page.get("fullurl", "")).strip()
            if not full_url and title:
                full_url = f"https://en.wikipedia.org/wiki/{urllib_parse.quote(title.replace(' ', '_'))}"
            extract = " ".join(str(page.get("extract", "")).split())
            content = extract or snippet
            if not title or not full_url or not content:
                continue
            dedupe_key = stable_hash(f"{full_url}|{content}")
            if dedupe_key in dedupe_keys:
                continue
            dedupe_keys.add(dedupe_key)
            results.append(
                WebDocument(
                    title=title,
                    url=full_url,
                    content=content,
                    score=round(max(0.35, 0.85 - ((rank - 1) * 0.08)), 2),
                    metadata={
                        "provider": self.provider_name,
                        "pageid": page_id,
                        "rank": rank,
                        "snippet": snippet[: self.config.web.snippet_chars],
                    },
                )
            )
            if len(results) >= result_limit:
                break

        return WebSearchResponse(
            provider=self.provider_name,
            query=query,
            results=tuple(results),
            degraded=bool(warnings),
            warnings=tuple(warnings),
            metadata={
                "attempt_count": max(1, attempt_count),
                "result_limit": result_limit,
                "search_hits": len(search_hits),
                "returned_results": len(results),
            },
        )

    def _request_json(self, params: dict[str, str]) -> dict[str, Any]:
        encoded_params = urllib_parse.urlencode(params)
        request_url = f"{self.config.web.api_base_url}?{encoded_params}"
        last_error: Exception | None = None
        attempts = self.config.web.max_retries + 1
        for attempt_index in range(attempts):
            try:
                return self._request_json_once(request_url)
            except (WebLookupError, WebLookupTimeoutError) as exc:
                last_error = exc
                if attempt_index >= attempts - 1:
                    break
                time.sleep(self.config.web.retry_backoff_s)
        if last_error is not None:
            raise last_error
        raise WebLookupError("web lookup failed without an error payload")

    def _request_json_once(self, request_url: str) -> dict[str, Any]:
        req = urllib_request.Request(
            request_url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.config.web.user_agent,
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=self.config.web.request_timeout_s) as response:
                payload = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise WebLookupTimeoutError(f"web lookup timed out for {request_url}") from exc
        except urllib_error.URLError as exc:
            reason = exc.reason if getattr(exc, "reason", None) else exc
            raise WebLookupError(f"web lookup unavailable for {request_url}: {reason}") from exc
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise WebLookupError(f"web lookup returned invalid JSON for {request_url}") from exc
        if not isinstance(decoded, dict):
            raise WebLookupError(f"web lookup returned an unexpected payload for {request_url}")
        return decoded

