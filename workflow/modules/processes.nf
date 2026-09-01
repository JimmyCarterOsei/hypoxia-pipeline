// Process definitions for the hypoxiapipe workflow.
//
// Each process is one CLI invocation, containerised independently. The
// container directive is per process, not global, which is the reason the
// images were split in Phase 6: BUILD_COHORT and CROSS_VALIDATE run the Python
// image, VALIDATE_SIGNATURE reaches the R image through the stdin/stdout
// contract, and neither needs the other's dependencies.

nextflow.enable.dsl = 2

// Build one cohort from its spec: fetch, harmonise, derive endpoint, QC, save.
//
// `publishDir` copies rather than symlinks so the published cohort survives a
// `work/` cleanup - a published artefact that evaporates with the scratch
// directory is worse than no artefact.
process BUILD_COHORT {
    tag "${cohort_name}"
    label 'process_medium'
    container params.python_image
    publishDir "${params.outdir}/cohorts", mode: 'copy', overwrite: true

    input:
    tuple val(cohort_name), path(spec)
    path cache

    output:
    tuple val(cohort_name), path("${cohort_name}"), emit: cohort
    path "${cohort_name}/qc.md",                    emit: qc
    path "${cohort_name}/manifest.json",            emit: manifest

    script:
    // params.signatures is a comma-separated String; .collect on a String
    // iterates its characters, so it must be tokenized first.
    def signatures = params.signatures.tokenize(',')*.trim().findAll()
        .collect { "--signature ${it}" }.join(' ')
    def offline    = params.offline ? '--offline' : ''
    def lenient    = params.lenient_expectations ? '--lenient' : ''
    // --no-fail-on-qc: a QC failure is recorded in the report and the run
    // continues, because deciding what to do about a failing cohort is the
    // analyst's call, not the scheduler's. The report is published either way.
    """
    export HYPOXIAPIPE_CACHE=\$(readlink -f ${cache})
    hypoxiapipe cohort build ${spec} \\
        --out ${cohort_name} \\
        ${offline} ${lenient} ${signatures} \\
        --no-fail-on-qc
    """
}

// Score signatures and fit Cox models. The estimation itself happens in R.
process VALIDATE_SIGNATURE {
    tag "${cohort_name}:${action}"
    label 'process_low'
    container params.python_image
    publishDir "${params.outdir}/validation", mode: 'copy', overwrite: true

    input:
    tuple val(cohort_name), path(cohort_dir), val(action)

    output:
    tuple val(cohort_name), val(action), path("${cohort_name}.${action}.tsv"), emit: results

    script:
    // params.signatures is a comma-separated String; .collect on a String
    // iterates its characters, so it must be tokenized first.
    def signatures = params.signatures.tokenize(',')*.trim().findAll()
        .collect { "--signature ${it}" }.join(' ')
    """
    hypoxiapipe model validate ${cohort_dir} \\
        ${signatures} \\
        --action ${action} \\
        --out ${cohort_name}.${action}.tsv
    """
}

// Nested cross-validation with preprocessing frozen inside every fold.
process CROSS_VALIDATE {
    tag "${cohort_name}:${estimator}"
    label 'process_high'
    container params.python_image
    publishDir "${params.outdir}/cv", mode: 'copy', overwrite: true

    input:
    tuple val(cohort_name), path(cohort_dir), val(estimator)

    output:
    path "${cohort_name}.${estimator}/*", emit: reports

    script:
    """
    hypoxiapipe model cv ${cohort_dir} \\
        --estimator ${estimator} \\
        --outer ${params.cv_outer} \\
        --inner ${params.cv_inner} \\
        --seed ${params.seed} \\
        --out ${cohort_name}.${estimator}
    """
}

// Re-hash every published manifest's outputs. Runs last, so the summary
// reports on artefacts as they were actually published rather than as they
// were when each process finished.
process VERIFY_MANIFESTS {
    label 'process_low'
    container params.python_image
    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
    path cohort_dirs, stageAs: 'cohort_*'

    output:
    path 'verification.md', emit: report

    script:
    // Each staged directory is named cohort_N, which says nothing useful in a
    // report covering four cohorts, so the real name is read back out of the
    // cohort metadata the build wrote.
    """
    : > verification.md
    for d in cohort_*; do
        name=\$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "\$d/cohort.json")
        echo "## \$name" >> verification.md
        hypoxiapipe manifest verify "\$d" >> verification.md || echo "FAILED: \$name" >> verification.md
        echo >> verification.md
    done
    """
}

// Collect the per-cohort validation tables into one file.
process COLLECT_RESULTS {
    label 'process_low'
    container params.python_image
    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    // Staged under their own names, not result_1..N: the filename carries the
    // cohort and action, and a combined table you cannot attribute to a cohort
    // is not a result.
    input:
    path tables

    output:
    path 'results.tsv', emit: table

    script:
    """
    python - <<'PY'
    import glob
    import pandas as pd

    frames = []
    for path in sorted(glob.glob("*.tsv")):
        if path == "results.tsv":
            continue
        cohort, action = path.rsplit(".", 2)[:2]
        frame = pd.read_csv(path, sep="\\t")
        frame.insert(0, "cohort", cohort)
        frame.insert(1, "action", action)
        frames.append(frame)
    if not frames:
        raise SystemExit("no validation tables to collect")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["cohort", "action", "name"])
    combined.to_csv("results.tsv", sep="\\t", index=False)
    print(f"combined {len(frames)} tables -> {combined.shape[0]} rows")
    PY
    """
}
