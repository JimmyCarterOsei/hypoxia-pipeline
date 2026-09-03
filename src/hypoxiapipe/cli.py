"""hypoxiapipe command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
import yaml

from hypoxiapipe.errors import HypoxiapipeError, SignatureError
from hypoxiapipe.signatures import registry

app = typer.Typer(no_args_is_help=True, help="Reproducible signature scoring.")
sig_app = typer.Typer(no_args_is_help=True, help="Signature registry commands.")
cohort_app = typer.Typer(no_args_is_help=True, help="Cohort ingest, QC and harmonisation.")
manifest_app = typer.Typer(no_args_is_help=True, help="Run manifests and provenance.")
model_app = typer.Typer(no_args_is_help=True, help="Survival modelling and validation.")
reference_app = typer.Typer(
    no_args_is_help=True, help="Scoring references for single-sample scoring."
)
app.add_typer(sig_app, name="sig")
app.add_typer(cohort_app, name="cohort")
app.add_typer(manifest_app, name="manifest")
app.add_typer(model_app, name="model")
app.add_typer(reference_app, name="reference")


@sig_app.command("list")
def sig_list() -> None:
    """List bundled signatures and their verification status."""
    for name, item in registry.list_bundled().items():
        if isinstance(item, registry.Signature):
            kind = "weighted" if item.weighted else "unweighted"
            typer.echo(f"  {name:<12} {item.n_genes:>3} genes  {kind:<10} {item.checksum[:19]}  OK")
        else:
            typer.echo(f"  {name:<12} INVALID: {type(item).__name__}")


@sig_app.command("verify")
def sig_verify(spec: Path) -> None:
    """Verify one spec file's checksum; non-zero exit on failure."""
    try:
        s = registry.load_spec(spec)
    except SignatureError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK: {s.name} ({s.n_genes} genes) {s.checksum}")


@sig_app.command("hash")
def sig_hash(
    spec: Path, write: bool = typer.Option(False, help="Write checksum into the file.")
) -> None:
    """Compute the canonical checksum for a spec (after source verification)."""
    raw = yaml.safe_load(spec.read_text())
    genes = raw.get("genes") or []
    if not genes:
        typer.echo("gene list is empty - fill it from the verified source first", err=True)
        raise typer.Exit(code=1)
    checksum = registry.compute_checksum([str(g) for g in genes], raw.get("coefficients"))
    typer.echo(checksum)
    if write:
        raw["checksum"] = checksum
        spec.write_text(yaml.safe_dump(raw, sort_keys=False, width=100))
        typer.echo(f"written to {spec}")


@app.command()
def score(
    matrix: Path = typer.Argument(..., help="TSV/CSV, genes as rows, samples as columns."),
    signature: str = typer.Option(..., "--signature", "-s", help="Bundled name or spec path."),
    method: str | None = typer.Option(None, help="Override the signature's default method."),
    out: Path | None = typer.Option(None, help="Write scores TSV here (default: stdout)."),
) -> None:
    """Score all samples in an expression matrix against a signature."""
    from hypoxiapipe.scoring import score as run_score

    sep = "," if matrix.suffix == ".csv" else "\t"
    df = pd.read_csv(matrix, sep=sep, index_col=0)

    spec_path = Path(signature)
    sig = registry.load_spec(spec_path) if spec_path.exists() else registry.load_bundled(signature)

    result = run_score(df, sig, method=method)
    typer.echo(
        f"# {result.signature} [{result.method}] genes {result.n_found}/{result.n_total} "
        f"(coverage {result.coverage:.0%}) checksum {result.checksum[:19]}",
        err=True,
    )
    if result.missing:
        typer.echo(f"# missing: {', '.join(result.missing)}", err=True)
    text = result.scores.to_csv(sep="\t", header=["score"])
    if out:
        out.write_text(text)
        typer.echo(f"# written {out}", err=True)
    else:
        typer.echo(text)


@cohort_app.command("list")
def cohort_list() -> None:
    """List bundled cohort specs and their pinned expectations."""
    from hypoxiapipe.ingest.spec import CohortSpec, list_bundled_cohorts

    for name, item in list_bundled_cohorts().items():
        if isinstance(item, CohortSpec):
            endpoint = item.endpoint.name if item.endpoint else "no endpoint"
            expected = item.expect.n_samples
            expected_txt = f"n={expected}" if expected else "n unpinned"
            typer.echo(
                f"  {name:<12} {item.source:<6} {item.accession or item.path:<12} "
                f"{item.platform or '-':<10} {endpoint:<22} {expected_txt}"
            )
        else:
            typer.echo(f"  {name:<12} INVALID: {item}")


@cohort_app.command("inspect")
def cohort_inspect(
    spec: str = typer.Argument(..., help="Bundled cohort name or spec path."),
    cache_dir: Path | None = typer.Option(
        None, help="Download cache directory [default: $HYPOXIAPIPE_CACHE]."
    ),
    offline: bool = typer.Option(False, help="Fail instead of downloading."),
) -> None:
    """Fetch a cohort and print its clinical columns without building it.

    Use this before pinning an endpoint: GEO characteristic keys come from free
    text, so the column names in a spec must be checked against reality once.
    """
    from hypoxiapipe.ingest.cache import Cache, default_cache_dir
    from hypoxiapipe.ingest.geo import load_geo
    from hypoxiapipe.ingest.spec import load_bundled_cohort, load_cohort_spec

    path = Path(spec)
    cs = load_cohort_spec(path) if path.exists() else load_bundled_cohort(spec)
    if cs.source != "geo" or not cs.accession:
        typer.echo("inspect currently supports GEO cohorts only", err=True)
        raise typer.Exit(code=2)

    cache = Cache(cache_dir or default_cache_dir(), offline=offline)
    cohort = load_geo(cs.accession, cache, name=cs.name, min_samples=1)
    typer.echo(f"{cs.name}: {cohort.n_genes} rows x {cohort.n_samples} samples")
    typer.echo(f"platform: {cohort.provenance.platform}")
    typer.echo("clinical columns:")
    for col in cohort.clinical.columns:
        sample_values = cohort.clinical[col].dropna().astype(str).unique()[:3]
        typer.echo(f"  {col:<32} e.g. {', '.join(sample_values)}")


@cohort_app.command("build")
def cohort_build(
    spec: str = typer.Argument(..., help="Bundled cohort name or spec path."),
    out: Path = typer.Option(..., "--out", "-o", help="Directory to write the cohort into."),
    cache_dir: Path | None = typer.Option(
        None, help="Download cache directory [default: $HYPOXIAPIPE_CACHE]."
    ),
    offline: bool = typer.Option(False, help="Fail instead of downloading."),
    signature: list[str] = typer.Option([], "--signature", "-s", help="Check coverage for these."),
    lenient: bool = typer.Option(False, help="Warn instead of failing on expectation mismatch."),
    fail_on_qc: bool = typer.Option(True, help="Exit non-zero if QC fails."),
) -> None:
    """Build a cohort from its spec: fetch, harmonise, derive endpoint, QC, save."""
    from hypoxiapipe.ingest.cache import Cache, default_cache_dir
    from hypoxiapipe.ingest.pipeline import build_cohort
    from hypoxiapipe.ingest.spec import load_bundled_cohort, load_cohort_spec
    from hypoxiapipe.ingest.store import save_cohort
    from hypoxiapipe.provenance import RunManifest

    manifest = RunManifest(
        params={
            "spec": spec,
            "out": str(out),
            "offline": offline,
            "signatures": list(signature),
            "lenient": lenient,
        }
    )
    path = Path(spec)
    cs = load_cohort_spec(path) if path.exists() else load_bundled_cohort(spec)
    sigs = [
        registry.load_spec(Path(s)) if Path(s).exists() else registry.load_bundled(s)
        for s in signature
    ]

    try:
        result = build_cohort(
            cs,
            Cache(cache_dir or default_cache_dir(), offline=offline),
            signatures=sigs or None,
            strict_expectations=not lenient,
        )
    except HypoxiapipeError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    save_cohort(result.cohort, out, extra={"build": result.to_dict()})
    (out / "qc.md").write_text(result.qc.to_markdown())
    (out / "qc.json").write_text(result.qc.to_json())

    manifest.absorb_cohort(result)
    manifest.absorb_signatures(sigs)
    for artefact in ("expression.parquet", "clinical.parquet", "cohort.json", "qc.json"):
        manifest.add_output_file(out / artefact)
    manifest.close(command=f"cohort build {spec}")
    manifest.write(out)

    typer.echo(result.qc.to_markdown())
    typer.echo(f"\nwritten to {out} (run {manifest.run_id})")
    for problem in result.expectation_failures:
        typer.echo(f"EXPECTATION: {problem}", err=True)
    if fail_on_qc and not result.qc.ok:
        raise typer.Exit(code=1)


@cohort_app.command("qc")
def cohort_qc(
    directory: Path = typer.Argument(..., help="Directory written by 'cohort build'."),
    signature: list[str] = typer.Option([], "--signature", "-s", help="Check coverage for these."),
    fail_on_qc: bool = typer.Option(True, help="Exit non-zero if QC fails."),
) -> None:
    """Re-run QC over a stored cohort, verifying its checksum on load."""
    from hypoxiapipe.ingest.store import load_cohort
    from hypoxiapipe.qc import run_qc

    try:
        cohort = load_cohort(directory)
    except HypoxiapipeError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    sigs = [
        registry.load_spec(Path(s)) if Path(s).exists() else registry.load_bundled(s)
        for s in signature
    ]
    report = run_qc(cohort, signatures=sigs or None, event_col="event")
    typer.echo(report.to_markdown())
    if fail_on_qc and not report.ok:
        raise typer.Exit(code=1)


@model_app.command("validate")
def model_validate(
    directory: Path = typer.Argument(..., help="Cohort directory from 'cohort build'."),
    signature: list[str] = typer.Option([], "--signature", "-s", help="Signatures to test."),
    action: str = typer.Option("cox_persd", help="cox_persd, cox_multivariable or quartile."),
    out: Path | None = typer.Option(None, help="Write the results TSV here."),
) -> None:
    """Score signatures in a stored cohort and fit Cox models in R.

    Per-SD is the default because it is the like-for-like comparison across
    signatures on different scales; quartile estimates need the events to
    support them and are reported alongside, never instead.
    """
    from hypoxiapipe.ingest.store import load_cohort
    from hypoxiapipe.modeling import cox, results_frame
    from hypoxiapipe.scoring import score as run_score

    try:
        cohort = load_cohort(directory)
    except HypoxiapipeError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for column in ("time_months", "event"):
        if column not in cohort.clinical.columns:
            typer.echo(
                f"cohort has no '{column}' column; rebuild it with an endpoint in its spec",
                err=True,
            )
            raise typer.Exit(code=2)

    scores: dict[str, list[float]] = {}
    for name in signature:
        spec_path = Path(name)
        sig = registry.load_spec(spec_path) if spec_path.exists() else registry.load_bundled(name)
        result = run_score(cohort.expr, sig)
        typer.echo(
            f"# {sig.name} [{result.method}] coverage {result.coverage:.0%} "
            f"({result.n_found}/{result.n_total})",
            err=True,
        )
        scores[sig.name] = list(result.scores.to_numpy())

    if not scores:
        typer.echo("no signatures given; pass at least one --signature", err=True)
        raise typer.Exit(code=2)

    try:
        results = cox(
            cohort.clinical["time_months"],
            cohort.clinical["event"],
            scores,
            action=action,
        )
    except HypoxiapipeError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    frame = results_frame(results)
    typer.echo(frame.to_string(index=False))
    if out:
        frame.to_csv(out, sep="\t", index=False)
        typer.echo(f"# written {out}", err=True)


@model_app.command("cv")
def model_cv(
    directory: Path = typer.Argument(..., help="Cohort directory from 'cohort build'."),
    estimator: str = typer.Option("coxnet", help="coxnet or rsf."),
    outer: int = typer.Option(5, help="Outer CV folds."),
    inner: int = typer.Option(3, help="Inner CV folds for penalty selection."),
    seed: int = typer.Option(0, help="Seed for fold assignment."),
    genes: Path | None = typer.Option(None, help="Restrict to genes listed in this file."),
    out: Path | None = typer.Option(None, help="Directory for the report and manifest."),
) -> None:
    """Nested cross-validation with preprocessing frozen inside every fold."""
    from hypoxiapipe.ingest.store import load_cohort
    from hypoxiapipe.modeling import nested_cv
    from hypoxiapipe.provenance import RunManifest

    try:
        cohort = load_cohort(directory)
    except HypoxiapipeError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    features = cohort.expr
    if genes:
        wanted = [g.strip() for g in genes.read_text().split() if g.strip()]
        present = [g for g in wanted if g in features.index]
        typer.echo(f"# restricted to {len(present)}/{len(wanted)} listed genes", err=True)
        features = features.loc[present]

    manifest = RunManifest(
        params={
            "cohort": str(directory),
            "estimator": estimator,
            "outer": outer,
            "inner": inner,
            "seed": seed,
        }
    )
    try:
        result = nested_cv(
            features,
            cohort.clinical["time_months"],
            cohort.clinical["event"],
            estimator=estimator,
            outer_splits=outer,
            inner_splits=inner,
            seed=seed,
        )
    except HypoxiapipeError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.to_markdown())
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "cv.md").write_text(result.to_markdown())
        (out / "cv.json").write_text(json.dumps(result.to_dict(), indent=2))
        manifest.add_input(
            cohort.name,
            checksum=cohort.expr_checksum,
            kind="cohort",
            population_hash=cohort.population_hash,
        )
        manifest.add_step("nested_cv", **result.to_dict())
        for artefact in ("cv.md", "cv.json"):
            manifest.add_output_file(out / artefact)
        manifest.close(command=f"model cv {directory}")
        manifest.write(out)
        typer.echo(f"\nwritten to {out} (run {manifest.run_id})")


@reference_app.command("create")
def reference_create(
    directory: Path = typer.Argument(..., help="Cohort directory from 'cohort build'."),
    signature: str = typer.Option(..., "--signature", "-s", help="Signature to freeze."),
    reference_id: str | None = typer.Option(None, help="Defaults to <cohort>-<signature>."),
    out: Path | None = typer.Option(None, help="Reference directory."),
) -> None:
    """Freeze a cohort's per-gene statistics so single samples can be scored.

    Cohort-relative scoring has no meaning for one sample. Freezing a named
    population gives a fixed distribution to score against, and the reference
    records which cohort it came from so any score remains attributable.
    """
    from hypoxiapipe.api.references import build_reference, save_reference
    from hypoxiapipe.ingest.store import load_cohort

    try:
        cohort = load_cohort(directory)
        spec_path = Path(signature)
        sig = (
            registry.load_spec(spec_path)
            if spec_path.exists()
            else registry.load_bundled(signature)
        )
        reference = build_reference(
            reference_id or f"{cohort.name.lower()}-{sig.name}",
            cohort.expr,
            sig,
            cohort_name=cohort.name,
            population_hash=cohort.population_hash,
            method=sig.scoring,
        )
        path = save_reference(reference, out)
    except HypoxiapipeError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(reference.describe(), indent=2))
    typer.echo(f"\nwritten to {path}", err=True)


@reference_app.command("list")
def reference_list(
    directory: Path | None = typer.Option(None, help="Reference directory."),
) -> None:
    """List registered scoring references."""
    from hypoxiapipe.api.references import list_references, reference_dir

    refs = list_references(directory)
    if not refs:
        typer.echo(f"no references in {directory or reference_dir()}", err=True)
        return
    for ref in refs:
        typer.echo(
            f"  {ref.reference_id:<28} {ref.signature:<10} n={ref.n_train:<5} "
            f"genes={len(ref.genes):<4} {ref.cohort}"
        )


@manifest_app.command("show")
def manifest_show(
    directory: Path = typer.Argument(..., help="Directory containing manifest.json."),
    inputs_only: bool = typer.Option(False, help="Print only the recorded inputs."),
) -> None:
    """Summarise a run manifest: what went in, what came out, on which code."""
    from hypoxiapipe.provenance import RunManifest

    try:
        data = RunManifest.load(directory)
    except HypoxiapipeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    env = data.get("env", {})
    typer.echo(f"run {data['run_id']}  {data.get('command') or ''}")
    typer.echo(f"  started  {data['started_at']}   finished {data.get('finished_at')}")
    typer.echo(
        f"  code     hypoxiapipe {env.get('hypoxiapipe_version')} "
        f"python {env.get('python')} git {env.get('git_sha') or 'not a repo'}"
    )
    if env.get("container_digest"):
        typer.echo(f"  image    {env['container_digest']}")
    typer.echo(f"  manifest {data.get('manifest_hash')}")

    typer.echo("\ninputs:")
    for artefact in data.get("inputs", []):
        typer.echo(f"  [{artefact['kind']:<16}] {artefact['name']:<24} {artefact['checksum'][:26]}")
    if inputs_only:
        return
    typer.echo("\noutputs:")
    for artefact in data.get("outputs", []):
        typer.echo(f"  {Path(artefact['name']).name:<22} {artefact['checksum'][:26]}")
    typer.echo(f"\nsteps: {', '.join(s['action'] for s in data.get('steps', []))}")


@manifest_app.command("verify")
def manifest_verify(
    directory: Path = typer.Argument(..., help="Directory containing manifest.json."),
    base: Path | None = typer.Option(None, help="Relocate relative output paths here."),
) -> None:
    """Re-hash a manifest's outputs; non-zero exit if any is missing or changed."""
    from hypoxiapipe.provenance import verify_manifest

    try:
        result = verify_manifest(directory, base=base)
    except HypoxiapipeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.to_markdown())
    if not result.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
