"""Per-working-dir shared KVStore + executor.

All storage classes that share the same db file get one SQLite connection
and one serialising ThreadPoolExecutor.  Reference-counting ensures the
store is closed only after the last user has finalised.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from snkv import KVStore


@dataclass
class SharedStore:
    kv: KVStore
    executor: ThreadPoolExecutor
    ref_count: int = 0


_registry: dict[str, SharedStore] = {}
_lock = threading.Lock()


def acquire(db_path: str) -> SharedStore:
    """Return (or create) the shared store for *db_path*, incrementing refcount."""
    with _lock:
        if db_path not in _registry:
            kv = KVStore(db_path)
            ex = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"snkv_{os.path.basename(db_path)}",
            )
            _registry[db_path] = SharedStore(kv=kv, executor=ex)
        entry = _registry[db_path]
        entry.ref_count += 1
        return entry


def release(db_path: str) -> None:
    """Decrement refcount; close KVStore + executor when the last user releases."""
    with _lock:
        entry = _registry.get(db_path)
        if entry is None:
            return
        entry.ref_count -= 1
        if entry.ref_count <= 0:
            try:
                entry.kv.close()
            except Exception:
                pass
            entry.executor.shutdown(wait=False)
            del _registry[db_path]
