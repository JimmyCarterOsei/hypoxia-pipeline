# STAR-Smith20-Cox v1 model card

**Status:** Research use only. This is not a clinical device or a treatment decision tool.

## Purpose and population

The model ranks biochemical-recurrence hazard after prostatectomy in prostate
tumours processed on the locked GDC-compatible STAR TPM expression domain. It was
developed retrospectively in TCGA-PRAD: 483 expression-matched patients, with
97 biochemical-recurrence events after 60-month administrative censoring.

Input is `gdc_star_tpm_log2p1_hgnc`: GDC-compatible STAR TPM followed by
`log2(TPM + 1)` and hypoxiapipe HGNC-symbol harmonisation. All 20 fixed Smith20
genes are required. Microarray, RSEM, FPKM, DESeq2 VST, raw counts and arbitrary
normalised matrices are not validated input domains.

## Frozen model

Gene means and sample SDs, the Smith20 composite mean and sample SD, and the
Cox coefficient were fitted once on the 483-patient development cohort and are
stored in the checksum-verified JSON bundle. Smith20 is the mean of the 20 gene
z-scores, then standardised using the training composite statistics. A
univariable Cox PH model with Efron ties uses beta **0.5806182816932**.
Inference never refits these quantities or rescales a submitted batch.

## Evidence

Repeated nested-CV held-out Harrell concordance for fixed Smith20 Cox: mean
**0.682540**, SD **0.063290**, median **0.697561**, patient-cluster bootstrap
95% CI **0.626876–0.736893**. These are internal discovery-cohort estimates,
not independent validation. Full-cohort fitted statistics are apparent-fit
descriptions and are recorded separately in `metrics.json`.

MSKCC, Cambridge and Stockholm reproduce the **cohort-relative Smith20
association** in their own legacy expression domains. **This is not independent
external validation of the frozen STAR-native predictor.** No compatible
independent RNA-seq recurrence cohort has yet validated its individual predictions.

## Outputs and limits

The API and CLI return standardised Smith20 score, Cox linear predictor,
relative hazard (`exp(linear_predictor)`) relative to score zero, and percentile
in the frozen TCGA development score distribution. They include the model ID,
version, gene/signature checksum, scaler checksum, population hash and manifest
checksum. They do not provide a calibrated absolute recurrence probability, a
clinical decision threshold, a risk category or a treatment recommendation.

Limitations include retrospective TCGA development, lack of independent
STAR-compatible predictor validation, and possible proportional-hazards
deviation: the full STAR 483/97 Smith20 analysis had Schoenfeld PH P≈0.0756,
while the shared 431/92 STAR analysis had P≈0.0280. The one-variable Cox form
is a prespecified research candidate, not a claim that hazards are constant in
every population or time window.

## Reproducibility and deployment

The immutable bundle is `models/prad-bcr-star-smith20-cox-v1/`. Its eight JSON
files carry a manifest and cross-file checksums. `HYPOXIAPIPE_MODELS=models`
registers the bundle for `GET /models`, `GET /models/{model_id}` and
`POST /predict/recurrence`. The input contract, response limitation and
research-use status are also exposed in OpenAPI `/docs`. Deployment image
metadata records the software version, Git revision, base image and image tag
outside the scientific bundle.
