"""Survival modelling: frozen preprocessing, penalised Cox, RSF, nested CV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypoxiapipe.modeling.preprocessing import PreprocessingError, ReferenceScaler
from hypoxiapipe.modeling.rbridge import (
    RBridgeError,
    SurvivalResult,
    cox,
    r_available,
    results_frame,
)

if TYPE_CHECKING:
    from hypoxiapipe.modeling.train import (
        CVResult,
        FittedModel,
        FoldResult,
        ModelingError,
        fit_final,
        nested_cv,
    )


def __getattr__(name: str) -> Any:
    """Load training-only scikit-survival dependencies only when requested."""
    if name in {"CVResult", "FittedModel", "FoldResult", "ModelingError", "fit_final", "nested_cv"}:
        from hypoxiapipe.modeling import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CVResult",
    "FittedModel",
    "FoldResult",
    "ModelingError",
    "PreprocessingError",
    "RBridgeError",
    "ReferenceScaler",
    "SurvivalResult",
    "cox",
    "fit_final",
    "nested_cv",
    "r_available",
    "results_frame",
]
