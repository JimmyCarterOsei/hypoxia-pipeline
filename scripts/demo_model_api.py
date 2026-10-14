#!/usr/bin/env python3
"""Short live API demo and optional known-fixture container smoke check."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "prad-bcr-star-smith20-cox-v1"
KNOWN = {
    "smith20_score": -0.9994416998626602,
    "linear_predictor": -0.5802941224267887,
    "relative_hazard": 0.5597337121145431,
    "training_percentile": 15.320910973084887,
}


def _request(
    base: str, route: str, payload: dict[str, Any] | None = None
) -> dict[str, Any] | list[Any]:
    encoded = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=encoded,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _fixture(known: bool) -> dict[str, Any]:
    if not known:
        return json.loads((ROOT / "docs/examples/recurrence-prediction.json").read_text())
    raw = json.loads((ROOT / "tests/fixtures/prad_bcr_star_v1_known_sample.json").read_text())
    return {
        "model_id": MODEL_ID,
        "expression_type": "gdc_star_tpm_log2p1_hgnc",
        "matrix": {
            "genes": list(raw["gene_values"]),
            "samples": [raw["patient"]],
            "values": [[value] for value in raw["gene_values"].values()],
        },
    }


def main() -> None:
    """Show the model and prove one patient's prediction is batch-invariant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--known-fixture", action="store_true", help="Run numerical regression.")
    parser.add_argument("--output", type=Path, help="Write machine-readable demo result JSON.")
    args = parser.parse_args()
    for attempt in range(30):
        try:
            health = _request(args.url, "/health")
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 29:
                raise
            time.sleep(1)
    print("1. Health:", json.dumps(health))
    models = _request(args.url, "/models")
    assert isinstance(models, list) and any(row["model_id"] == MODEL_ID for row in models)
    print("2. Models:", json.dumps(models))
    detail = _request(args.url, f"/models/{MODEL_ID}")
    assert isinstance(detail, dict) and detail["research_use_only"] is True
    print("3. Model:", detail["model_id"], detail["version"], detail["expected_expression_type"])

    request = _fixture(args.known_fixture)
    single = _request(args.url, "/predict/recurrence", request)
    assert isinstance(single, dict) and single["research_use_only"] is True
    assert "recurrence_probability" not in json.dumps(single)
    patient = single["predictions"][0]
    print("4. Single prediction:", json.dumps(patient))
    print("5. Provenance:", json.dumps(single["provenance"]))
    if args.known_fixture:
        for field, expected in KNOWN.items():
            assert abs(patient[field] - expected) <= 1e-12, (field, patient[field], expected)

    batch_request = json.loads(json.dumps(request))
    batch_request["matrix"]["samples"].append("demo-other")
    for row in batch_request["matrix"]["values"]:
        row.append(row[0] + 0.1)
    batch = _request(args.url, "/predict/recurrence", batch_request)
    assert isinstance(batch, dict)
    assert batch["predictions"][0] == patient
    print("6. Batch identity: PASS")
    if args.output:
        args.output.write_text(
            json.dumps(
                {"health": health, "models": models, "single": single, "batch": batch}, indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
