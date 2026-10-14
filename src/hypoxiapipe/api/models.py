"""Explicit, integrity-checked registry for frozen research prediction bundles."""

from __future__ import annotations

import os
from pathlib import Path

from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.modeling.bundle import BundleError, Smith20CoxBundle, load_model_bundle

MODELS_DIR_ENV = "HYPOXIAPIPE_MODELS"


class ModelRegistryError(HypoxiapipeError):
    """An operator-configured model bundle failed integrity verification."""


def list_models() -> dict[str, Smith20CoxBundle]:
    """Load only immediate children of an explicitly configured model directory."""
    configured = os.environ.get(MODELS_DIR_ENV)
    if not configured:
        return {}
    root = Path(configured)
    if not root.is_dir():
        raise ModelRegistryError("configured model directory is missing or unreadable")
    loaded: dict[str, Smith20CoxBundle] = {}
    try:
        children = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError as exc:
        raise ModelRegistryError("configured model directory is unreadable") from exc
    for path in children:
        try:
            bundle = load_model_bundle(path)
        except (BundleError, OSError, KeyError, TypeError, ValueError) as exc:
            raise ModelRegistryError(
                f"configured bundle {path.name!r} failed integrity verification"
            ) from exc
        if path.name != bundle.model_id or bundle.model_id in loaded:
            raise ModelRegistryError(f"configured bundle {path.name!r} has invalid identity")
        loaded[bundle.model_id] = bundle
    return loaded
