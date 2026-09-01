# syntax=docker/dockerfile:1.7
###############################################################################
# hypoxiapipe - R survival stage
#
# One job: read a JSON request on stdin, fit the survival model, write a JSON
# response on stdout. Nothing else is installed and nothing is exposed.
#
#   docker run --rm -i ghcr.io/JimmyCarterOsei/hypoxiapipe-r:TAG < request.json
#
# From the Python side, set:
#   HYPOXIAPIPE_R_COMMAND="docker run --rm -i ghcr.io/JimmyCarterOsei/hypoxiapipe-r:TAG"
#
# rocker/r-ver pins the R version *and* an RSPM snapshot date, so package
# versions are fixed by the base image rather than by whatever CRAN served on
# build day. That is the whole reason to use it over r-base.
###############################################################################

FROM rocker/r-ver:4.3.3

LABEL org.opencontainers.image.title="hypoxiapipe-r" \
      org.opencontainers.image.description="Survival estimation worker (R survival package)" \
      org.opencontainers.image.source="https://github.com/JimmyCarterOsei/hypoxia-pipeline" \
      org.opencontainers.image.licenses="MIT"

# `survival` ships with R; installing it explicitly pins it via the snapshot.
RUN install2.r --error --skipinstalled survival jsonlite \
 && rm -rf /tmp/downloaded_packages

COPY src/hypoxiapipe/modeling/r/survival_worker.R /opt/hypoxiapipe/survival_worker.R

RUN useradd --create-home --uid 1000 worker \
 && chown -R worker:worker /opt/hypoxiapipe
USER worker
WORKDIR /home/worker

# Verify the worker answers correctly at build time. A request with too few
# observations must be *refused*, so this checks the failure path as well as
# the startup path - the response must be a well-formed JSON refusal, not a
# crash and not a fitted model.
RUN echo '{"action":"cox_persd","time":[1],"event":[1],"scores":{"s":[1]}}' \
      | Rscript --vanilla /opt/hypoxiapipe/survival_worker.R \
      | grep -q '"ok":false' \
    || (echo "worker did not refuse an undersized request" && exit 1)

ENTRYPOINT ["Rscript", "--vanilla", "/opt/hypoxiapipe/survival_worker.R"]
