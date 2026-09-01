"""Hashing helpers shared by the provenance layer.

Everything the pipeline records is identified by content, not by filename or
timestamp. A path tells you where a file was; a hash tells you whether it is
the file the result was computed from.

All digests are SHA-256 and carry a ``sha256:`` prefix so a bare hex string can
never be mistaken for one, and so the algorithm can change later without any
stored value becoming ambiguous.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PREFIX = "sha256"
_CHUNK = 1 << 20


def _tag(digest: str) -> str:
    return f"{PREFIX}:{digest}"


def hash_bytes(data: bytes) -> str:
    """Return the tagged SHA-256 of a byte string."""
    return _tag(hashlib.sha256(data).hexdigest())


def hash_text(text: str) -> str:
    """Return the tagged SHA-256 of a UTF-8 string."""
    return hash_bytes(text.encode())


def hash_file(path: str | Path) -> str:
    """Return the tagged SHA-256 of a file, streamed rather than read whole."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return _tag(h.hexdigest())


def hash_json(obj: Any) -> str:
    """Return the tagged SHA-256 of a JSON-serialisable object.

    Keys are sorted and separators fixed so that two structurally identical
    objects hash the same regardless of construction order.
    """
    return hash_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def hash_directory(path: str | Path, pattern: str = "*") -> dict[str, str]:
    """Return ``{relative_path: hash}`` for every matching file in a directory."""
    root = Path(path)
    return {
        str(p.relative_to(root)): hash_file(p) for p in sorted(root.rglob(pattern)) if p.is_file()
    }
