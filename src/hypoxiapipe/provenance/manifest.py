"""Run manifests.

A manifest answers one question about any number this pipeline produced: what
exactly went into it. Not approximately — the accession *and* the checksum of
the bytes downloaded, the signature *and* its content hash, the symbol
authority *and* the hash of the mapping table, the code version, the parameters,
and the hash of every file written.

That matters because the failure this project exists to prevent was not a bug
in anyone's code. It was an input that quietly stopped being what its label
said. A result with a manifest can be re-checked against its inputs a year
later; a result without one has to be taken on trust.

``RunManifest.verify()`` re-hashes the recorded outputs and reports drift, so a
stale result on disk is detectable rather than assumed current.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypoxiapipe import __version__
from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.provenance.hashing import hash_file, hash_json

MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA = 1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_sha(repo: str | Path | None = None) -> str | None:
    """Return the current git commit, with ``-dirty`` appended for a dirty tree.

    Returns ``None`` outside a repository rather than raising: a run from an
    installed wheel is legitimate, and 'not under version control' is itself
    useful information to record.
    """
    cwd = str(repo) if repo else None
    try:
        sha = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return f"{sha}-dirty" if dirty else sha


def environment() -> dict[str, Any]:
    """Capture the execution environment, including any container digest."""
    return {
        "hypoxiapipe_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": git_sha(),
        # Set by the container entrypoint in Phase 6; absent when running locally.
        "container_digest": os.environ.get("HYPOXIAPIPE_IMAGE_DIGEST"),
    }


@dataclass(frozen=True)
class Artefact:
    """One hashed thing a run consumed or produced."""

    role: str
    name: str
    checksum: str
    kind: str = "file"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    """Everything needed to explain, and later re-check, one run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    command: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=environment)
    inputs: list[Artefact] = field(default_factory=list)
    outputs: list[Artefact] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    schema: int = MANIFEST_SCHEMA

    # -- recording --------------------------------------------------------
    def add_input(self, name: str, checksum: str, kind: str, **detail: Any) -> None:
        """Record a consumed artefact by content hash."""
        self.inputs.append(
            Artefact(role="input", name=name, checksum=checksum, kind=kind, detail=detail)
        )

    def add_output_file(self, path: str | Path, **detail: Any) -> None:
        """Hash a written file and record it."""
        p = Path(path)
        self.outputs.append(
            Artefact(
                role="output",
                name=str(p),
                checksum=hash_file(p),
                kind="file",
                detail=detail,
            )
        )

    def add_step(self, action: str, **detail: Any) -> None:
        """Append a timestamped processing step."""
        self.steps.append({"action": action, "at": _now(), **detail})

    def absorb_cohort(self, build: Any) -> None:
        """Record the inputs, steps and identity of a cohort build.

        Takes a ``BuildResult``; kept untyped to avoid a provenance -> ingest
        import cycle, since ingest already writes manifests.
        """
        cohort = build.cohort
        prov = cohort.provenance
        if prov.accession:
            self.add_input(
                prov.accession,
                checksum=cohort.expr_checksum,
                kind="cohort",
                source=prov.source,
                url=prov.url,
                platform=prov.platform,
                retrieved_at=prov.retrieved_at,
                n_samples=cohort.n_samples,
                n_genes=cohort.n_genes,
                population_hash=cohort.population_hash,
            )
        if prov.symbol_authority:
            authority_hash = next(
                (
                    s.detail.get("authority_checksum")
                    for s in prov.steps
                    if s.action == "harmonise_symbols"
                ),
                None,
            )
            self.add_input(
                prov.symbol_authority,
                checksum=authority_hash or "unrecorded",
                kind="symbol_authority",
            )
        for step in prov.steps:
            self.add_step(step.action, **step.detail)

    def absorb_signatures(self, signatures: list[Any]) -> None:
        """Record each signature by its registry checksum."""
        for sig in signatures:
            self.add_input(
                sig.name,
                checksum=sig.checksum,
                kind="signature",
                n_genes=sig.n_genes,
                scoring=sig.scoring,
                source=sig.source,
            )

    def close(self, command: str | None = None) -> RunManifest:
        """Stamp the finish time and return self, for chaining."""
        self.finished_at = _now()
        if command:
            self.command = command
        return self

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain dictionary, with its own content hash."""
        body = {
            "schema": self.schema,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "command": self.command,
            "params": self.params,
            "env": self.env,
            "inputs": [asdict(a) for a in self.inputs],
            "outputs": [asdict(a) for a in self.outputs],
            "steps": self.steps,
        }
        # Self-hash covers inputs and parameters but not outputs, which are
        # appended as the run proceeds; it identifies "this run, so configured".
        body["manifest_hash"] = hash_json(
            {k: v for k, v in body.items() if k not in {"outputs", "finished_at"}}
        )
        return body

    def to_json(self) -> str:
        """Return the manifest as indented JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def write(self, directory: str | Path, filename: str = MANIFEST_FILE) -> Path:
        """Write the manifest into a directory and return its path."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        path.write_text(self.to_json())
        return path

    # -- checking ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Load a manifest as a dictionary (manifests are read, not resumed)."""
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_FILE
        if not p.exists():
            raise HypoxiapipeError(f"no manifest at {p}")
        loaded: dict[str, Any] = json.loads(p.read_text())
        return loaded


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of re-checking a manifest's outputs against what is on disk."""

    manifest: str
    checked: int
    missing: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every recorded output is present and unchanged."""
        return not self.missing and not self.changed

    def to_markdown(self) -> str:
        """Render a short human-readable verdict."""
        lines = [f"# Manifest verification - {self.manifest}", ""]
        lines.append(f"- outputs checked: {self.checked}")
        for name in self.missing:
            lines.append(f"- x MISSING: {name}")
        for name in self.changed:
            lines.append(f"- x CHANGED: {name}")
        lines += ["", f"**Result: {'PASS' if self.ok else 'FAIL'}**"]
        return "\n".join(lines)


def verify_manifest(path: str | Path, base: str | Path | None = None) -> VerifyResult:
    """Re-hash a manifest's recorded outputs and report anything that drifted.

    ``base`` relocates relative output paths, so a manifest written on a
    compute node can be verified against artefacts pulled back locally.
    """
    p = Path(path)
    manifest_path = p / MANIFEST_FILE if p.is_dir() else p
    data = RunManifest.load(manifest_path)
    root = Path(base) if base else manifest_path.parent

    missing: list[str] = []
    changed: list[str] = []
    outputs = data.get("outputs", [])
    for artefact in outputs:
        name = artefact.get("name", "")
        candidate = Path(name)
        if not candidate.exists():
            # Artefacts recorded on one machine are often verified on another,
            # so fall back to the same filename under `base` before giving up.
            candidate = root / candidate.name
        if not candidate.exists():
            missing.append(name)
            continue
        if hash_file(candidate) != artefact.get("checksum"):
            changed.append(name)

    return VerifyResult(
        manifest=str(manifest_path),
        checked=len(outputs),
        missing=tuple(missing),
        changed=tuple(changed),
    )
