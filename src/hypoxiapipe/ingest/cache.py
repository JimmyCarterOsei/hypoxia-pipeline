"""On-disk cache for downloaded cohort files.

Every fetch is content-hashed and written atomically alongside a sidecar
``.meta.json`` recording the URL, retrieval time, byte count and SHA-256. Two
consequences matter:

* **CI never hits the network.** Tests run with ``offline=True`` against
  committed fixtures; a cache miss is an error rather than a silent download.
* **A run is reproducible against a specific byte stream.** The manifest
  (Phase 3) cites the file checksum, not just the accession, so "we used
  GSE70768" becomes "we used these exact bytes of GSE70768".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from hypoxiapipe.errors import CacheMissError, DownloadError

DEFAULT_CACHE = Path(os.environ.get("HYPOXIAPIPE_CACHE", "~/.cache/hypoxiapipe")).expanduser()


def default_cache_dir() -> Path:
    """Return the cache directory, reading the environment at call time.

    ``DEFAULT_CACHE`` is bound at import, which is wrong for a CLI whose
    environment is set by the caller (a container entrypoint, a Nextflow
    process) after the module is first imported.
    """
    return Path(os.environ.get("HYPOXIAPIPE_CACHE", "~/.cache/hypoxiapipe")).expanduser()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


@dataclass(frozen=True)
class CacheEntry:
    """A cached file and its recorded metadata."""

    path: Path
    checksum: str
    url: str | None
    retrieved_at: str
    n_bytes: int


class Cache:
    """A simple keyed file cache rooted at a directory."""

    def __init__(self, root: Path | str = DEFAULT_CACHE, offline: bool = False) -> None:
        """Create a cache rooted at `root`; `offline` disables downloads."""
        self.root = Path(root).expanduser()
        self.offline = offline

    def path_for(self, key: str) -> Path:
        """Return the filesystem path for a cache key (keys may contain '/')."""
        safe = quote(key, safe="/")
        return self.root / safe

    def meta_for(self, key: str) -> Path:
        """Return the path of the sidecar metadata file for a key."""
        return self.path_for(key).with_suffix(self.path_for(key).suffix + ".meta.json")

    def has(self, key: str) -> bool:
        """Return True if the key is present in the cache."""
        return self.path_for(key).is_file()

    def get(self, key: str) -> CacheEntry:
        """Return a cached entry, or raise ``CacheMissError``."""
        path = self.path_for(key)
        if not path.is_file():
            raise CacheMissError(
                f"'{key}' is not in the cache at {self.root}. "
                "Run the ingest step online once to populate it, or point "
                "HYPOXIAPIPE_CACHE at a directory that has it."
            )
        meta_path = self.meta_for(key)
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return CacheEntry(
            path=path,
            checksum=meta.get("checksum") or sha256_file(path),
            url=meta.get("url"),
            retrieved_at=meta.get("retrieved_at", "unknown"),
            n_bytes=path.stat().st_size,
        )

    def put(self, key: str, data: bytes, url: str | None = None) -> CacheEntry:
        """Write bytes into the cache atomically and record metadata."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(path))
        entry = CacheEntry(
            path=path,
            checksum=sha256_file(path),
            url=url,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            n_bytes=path.stat().st_size,
        )
        self.meta_for(key).write_text(
            json.dumps(
                {
                    "key": key,
                    "url": entry.url,
                    "checksum": entry.checksum,
                    "retrieved_at": entry.retrieved_at,
                    "n_bytes": entry.n_bytes,
                },
                indent=2,
            )
        )
        return entry

    def fetch(
        self,
        key: str,
        url: str,
        downloader: Callable[[str], bytes] | None = None,
        timeout: int = 120,
    ) -> CacheEntry:
        """Return a cached entry, downloading it once if absent.

        In offline mode a miss raises rather than downloading, which is what
        keeps CI deterministic.
        """
        if self.has(key):
            return self.get(key)
        if self.offline:
            raise CacheMissError(
                f"offline mode: '{key}' not cached at {self.root} and downloads are disabled"
            )
        fn = downloader or _default_downloader
        try:
            data = fn(url) if downloader else fn(url, timeout)  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise DownloadError(f"failed to download {url}: {exc}") from exc
        return self.put(key, data, url=url)


def _default_downloader(url: str, timeout: int = 120) -> bytes:
    """Fetch a URL into memory (used only outside CI)."""
    with urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https/ftp hosts
        return bytes(resp.read())
