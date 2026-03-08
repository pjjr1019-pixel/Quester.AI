"""Bounded in-memory caches with explicit snapshots for local-AI control surfaces."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from data_structures import BoundedCacheSnapshot


class BoundedCache:
    """A tiny auditable LRU cache for one namespace."""

    def __init__(self, namespace: str, *, max_entries: int) -> None:
        if not namespace.strip():
            raise ValueError("namespace must not be empty.")
        if max_entries < 1:
            raise ValueError("max_entries must be positive.")
        self.namespace = namespace
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Any | None:
        if key not in self._entries:
            self._misses += 1
            return None
        self._hits += 1
        value = self._entries.pop(key)
        self._entries[key] = value
        return value

    def put(self, key: str, value: Any) -> None:
        if key in self._entries:
            self._entries.pop(key)
        self._entries[key] = value
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def snapshot(self) -> BoundedCacheSnapshot:
        return BoundedCacheSnapshot(
            namespace=self.namespace,
            max_entries=self.max_entries,
            entry_count=len(self._entries),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            warm_keys=tuple(self._entries.keys()),
        )

    def recent_items(self, *, limit: int | None = None) -> tuple[tuple[str, Any], ...]:
        """Return recent entries in most-recent-first order without exposing mutability."""
        items = list(self._entries.items())
        recent = list(reversed(items))
        if limit is not None:
            recent = recent[: max(1, int(limit))]
        return tuple((str(key), value) for key, value in recent)


class BoundedCacheManager:
    """Owns multiple small named caches used across local-AI subsystems."""

    def __init__(self, cache_limits: dict[str, int]) -> None:
        self._caches = {
            namespace: BoundedCache(namespace, max_entries=max_entries)
            for namespace, max_entries in cache_limits.items()
        }

    def get(self, namespace: str, key: str) -> Any | None:
        return self._require(namespace).get(key)

    def put(self, namespace: str, key: str, value: Any) -> None:
        self._require(namespace).put(key, value)

    def snapshot(self, namespace: str) -> BoundedCacheSnapshot:
        return self._require(namespace).snapshot()

    def snapshots(self) -> tuple[BoundedCacheSnapshot, ...]:
        return tuple(cache.snapshot() for cache in self._caches.values())

    def recent_items(self, namespace: str, *, limit: int | None = None) -> tuple[tuple[str, Any], ...]:
        return self._require(namespace).recent_items(limit=limit)

    def _require(self, namespace: str) -> BoundedCache:
        if namespace not in self._caches:
            raise KeyError(f"Unknown cache namespace: {namespace}")
        return self._caches[namespace]
