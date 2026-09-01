#!/usr/bin/env nextflow
///////////////////////////////////////////////////////////////////////////////
// hypoxiapipe - cohort ingest through survival validation.
//
//   nextflow run workflow/main.nf -profile test        # ~1 min, no network
//   nextflow run workflow/main.nf -profile docker --cohorts cambridge,tcga-prad
//
// Cohorts fan out as a channel: each is built, validated and cross-validated
// independently and in parallel, then the results are collected. That is the
// entire reason to use a workflow manager here rather than a shell script -
// four cohorts on a laptop and four cohorts on AWS Batch are the same DSL with
// a different executor.
///////////////////////////////////////////////////////////////////////////////

nextflow.enable.dsl = 2

include {
    BUILD_COHORT;
    VALIDATE_SIGNATURE;
    CROSS_VALIDATE;
    VERIFY_MANIFESTS;
    COLLECT_RESULTS;
} from './modules/processes.nf'

def helpMessage() {
    log.info """
    hypoxiapipe workflow

    Usage:
      nextflow run workflow/main.nf -profile test
      nextflow run workflow/main.nf -profile docker --cohorts cambridge,stockholm

    Options:
      --cohorts        Comma-separated cohort specs (bundled names or paths)
      --signatures     Comma-separated signatures to score
      --actions        Cox actions to run (cox_persd,quartile)
      --estimators     Models for nested CV (coxnet,rsf); empty to skip
      --outdir         Where published results go            [${params.outdir}]
      --cache          Download cache directory              [${params.cache}]
      --offline        Fail instead of downloading           [${params.offline}]
      --cv_outer       Outer CV folds                        [${params.cv_outer}]
      --seed           Seed for fold assignment              [${params.seed}]
    """.stripIndent()
}

workflow {
    if (params.help) {
        helpMessage()
        return
    }

    if (!params.cohorts) {
        error "No cohorts requested. Pass --cohorts, e.g. --cohorts cambridge,tcga-prad"
    }
    if (!params.signatures) {
        error "No signatures requested. Pass --signatures, e.g. --signatures smith20"
    }

    log.info """
    ${'='*70}
    hypoxiapipe  |  cohorts: ${params.cohorts}
                 |  signatures: ${params.signatures}
                 |  offline: ${params.offline}   outdir: ${params.outdir}
    ${'='*70}
    """.stripIndent()

    // A spec is either a bundled name or a path to a YAML file. Resolving that
    // here means the processes only ever see a real file.
    cohort_specs = Channel
        .fromList(params.cohorts.tokenize(','))
        .map { it.trim() }
        .filter { it }
        .map { name ->
            def local = file(name)
            def resolved = local.exists()
                ? local
                : file("${projectDir}/../src/hypoxiapipe/ingest/specs/${name}.yaml")
            if (!resolved.exists()) {
                error "Cohort spec not found: '${name}' (looked for ${local} and ${resolved})"
            }
            tuple(resolved.baseName, resolved)
        }

    cache_dir = file(params.cache)
    cache_dir.mkdirs()

    BUILD_COHORT(cohort_specs, cache_dir)

    // Every cohort x every requested Cox action.
    actions = Channel.fromList(params.actions.tokenize(',')).map { it.trim() }.filter { it }
    VALIDATE_SIGNATURE(BUILD_COHORT.out.cohort.combine(actions))

    COLLECT_RESULTS(VALIDATE_SIGNATURE.out.results.map { _c, _a, table -> table }.collect())

    // Nested CV is optional: on a 20-gene signature it is a comparison, not
    // the headline result, and it is much the most expensive step.
    if (params.estimators) {
        estimators = Channel
            .fromList(params.estimators.tokenize(','))
            .map { it.trim() }
            .filter { it }
        CROSS_VALIDATE(BUILD_COHORT.out.cohort.combine(estimators))
    }

    VERIFY_MANIFESTS(BUILD_COHORT.out.cohort.map { _name, dir -> dir }.collect())
}

workflow.onComplete {
    log.info """
    ${'='*70}
    ${workflow.success ? 'Completed' : 'FAILED'} in ${workflow.duration}
    Published to: ${params.outdir}
    Run name:     ${workflow.runName}
    Commit:       ${workflow.commitId ?: 'not a git checkout'}
    ${'='*70}
    """.stripIndent()
}
