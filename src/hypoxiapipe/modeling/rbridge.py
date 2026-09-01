"""The Python half of the polyglot survival contract.

Survival estimation is delegated to R's ``survival`` package over a JSON
request/response on stdin/stdout. The boundary is deliberately narrow: numbers
in, numbers out, no shared state, no file passing. That means the R worker can
be tested on its own, run in its own container, and swapped for a different
implementation without anything upstream changing.

The bridge treats R's absence as a clear, actionable error rather than silently
falling back to a Python approximation. A quiet fallback would mean two
different estimators producing results labelled identically - which is exactly
the class of problem this project exists to prevent.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hypoxiapipe.errors import HypoxiapipeError

ACTIONS = ("cox_persd", "cox_multivariable", "quartile")
DEFAULT_TIMEOUT = 300

#: Overrides the command used to reach the R worker. The contract is one JSON
#: request on stdin and one response on stdout, so anything that speaks that
#: works: a local Rscript, a container, or a remote shell.
#:
#:   HYPOXIAPIPE_R_COMMAND="docker run --rm -i ghcr.io/<org>/hypoxiapipe-r:0.1.0"
#:
#: This is what makes the split-container layout possible without a service,
#: a socket or a shared volume - the narrow contract from Phase 4 paying off.
R_COMMAND_ENV = "HYPOXIAPIPE_R_COMMAND"


class RBridgeError(HypoxiapipeError):
    """The R survival worker was unavailable, failed, or returned a bad response."""


def rscript_path() -> str | None:
    """Return the path to Rscript, or None when R is not installed."""
    return shutil.which("Rscript")


def worker_script() -> Path:
    """Return the path to the bundled R worker."""
    ref = resources.files("hypoxiapipe.modeling.r").joinpath("survival_worker.R")
    return Path(str(ref))


def r_command() -> list[str] | None:
    """Return the argv used to reach the R worker, or None if unreachable.

    An explicit ``HYPOXIAPIPE_R_COMMAND`` wins; the image it names is expected
    to have the worker as its entrypoint, so no script path is appended. With
    no override, a local ``Rscript`` runs the bundled worker directly.
    """
    override = os.environ.get(R_COMMAND_ENV, "").strip()
    if override:
        return shlex.split(override)
    exe = rscript_path()
    if exe is None:
        return None
    return [exe, "--vanilla", str(worker_script())]


def r_available() -> bool:
    """Return True when the R worker can be reached and has its dependencies."""
    if os.environ.get(R_COMMAND_ENV, "").strip():
        # A configured transport is taken at its word: probing it would mean
        # starting a container on every import.
        return True
    exe = rscript_path()
    if exe is None:
        return False
    probe = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            exe,
            "-e",
            'quit(status = as.integer(!(requireNamespace("survival", quietly=TRUE) '
            '&& requireNamespace("jsonlite", quietly=TRUE))))',
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return probe.returncode == 0


@dataclass(frozen=True)
class SurvivalResult:
    """One fitted survival estimate returned by the R worker."""

    name: str
    model: str
    hr: float
    ci_low: float
    ci_high: float
    p: float
    c_index: float
    n: int
    n_events: int
    detail: dict[str, Any]

    @property
    def significant(self) -> bool:
        """Return True when p < 0.05 and the hazard ratio exceeds 1."""
        return bool(self.p < 0.05 and self.hr > 1)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form for reports and manifests."""
        return {
            "name": self.name,
            "model": self.model,
            "hr": self.hr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p": self.p,
            "c_index": self.c_index,
            "n": self.n,
            "n_events": self.n_events,
            **self.detail,
        }


def _scalar_true(value: Any) -> bool:
    """Interpret a JSON flag that may arrive as a scalar or a length-1 array.

    R serialises length-1 vectors as arrays unless explicitly unboxed, and
    ``[false]`` is truthy in Python. Reading the flag defensively means a
    malformed response from the other side of the contract cannot be mistaken
    for success.
    """
    if isinstance(value, list):
        value = value[0] if value else False
    return bool(value)


def _clean(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float)]


def call_r(request: dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Send one request to the R worker and return its parsed response."""
    argv = r_command()
    if argv is None:
        raise RBridgeError(
            "no R worker available. Survival estimation runs in R by design: install R "
            "with the 'survival' and 'jsonlite' packages, or set "
            f'{R_COMMAND_ENV}="docker run --rm -i ghcr.io/<org>/hypoxiapipe-r:<tag>" '
            "to use the containerised stage."
        )
    try:
        proc = subprocess.run(  # noqa: S603 - argv from config, never a shell string
            argv,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RBridgeError(f"R worker timed out after {timeout}s") from exc
    except (OSError, FileNotFoundError) as exc:
        raise RBridgeError(f"could not start the R worker ({' '.join(argv)}): {exc}") from exc

    stdout = proc.stdout.strip()
    if not stdout:
        raise RBridgeError(
            f"R worker produced no output (exit {proc.returncode}) "
            f"running {' '.join(argv)}. stderr:\n{proc.stderr.strip()}"
        )
    try:
        payload: dict[str, Any] = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RBridgeError(
            f"R worker returned unparseable output (exit {proc.returncode}):\n{stdout[:500]}"
        ) from exc

    if not _scalar_true(payload.get("ok")):
        error = payload.get("error")
        if isinstance(error, list):
            error = error[0] if error else None
        raise RBridgeError(f"R worker refused the request: {error}")
    return payload


def _to_results(payload: dict[str, Any]) -> list[SurvivalResult]:
    out: list[SurvivalResult] = []
    for row in payload.get("results", []):
        if "error" in row:
            raise RBridgeError(f"{row.get('name')}: {row['error']}")
        known = {"name", "model", "hr", "ci_low", "ci_high", "p", "c_index", "n", "n_events"}
        out.append(
            SurvivalResult(
                name=str(row["name"]),
                model=str(row["model"]),
                hr=float(row["hr"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                p=float(row["p"]),
                c_index=float(row["c_index"]),
                n=int(row["n"]),
                n_events=int(row["n_events"]),
                detail={k: v for k, v in row.items() if k not in known},
            )
        )
    return out


def cox(
    time: Any,
    event: Any,
    scores: dict[str, Any],
    action: str = "cox_persd",
    covariates: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[SurvivalResult]:
    """Fit Cox models in R and return one result per score.

    Parameters
    ----------
    time : array-like
        Follow-up time in months, strictly positive.
    event : array-like
        Event indicator coded 0/1.
    scores : dict[str, array-like]
        One entry per signature score, all the same length as ``time``.
    action : str
        ``cox_persd`` (default), ``cox_multivariable`` or ``quartile``.
    covariates : dict[str, array-like] | None
        Additional adjustment variables, used by ``cox_multivariable``.
    timeout : int
        Seconds to wait for the R worker.

    """
    if action not in ACTIONS:
        raise RBridgeError(f"unknown action {action!r} (choose from {ACTIONS})")
    request: dict[str, Any] = {
        "action": action,
        "time": _clean(time),
        "event": _clean(event),
        "scores": {k: _clean(v) for k, v in scores.items()},
    }
    if covariates:
        request["covariates"] = {k: _clean(v) for k, v in covariates.items()}
    return _to_results(call_r(request, timeout=timeout))


def results_frame(results: list[SurvivalResult]) -> pd.DataFrame:
    """Return survival results as a tidy DataFrame for reporting."""
    return pd.DataFrame([r.to_dict() for r in results])
