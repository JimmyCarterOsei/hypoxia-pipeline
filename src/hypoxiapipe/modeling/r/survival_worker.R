#!/usr/bin/env Rscript
##############################################################################
# survival_worker.R
#
# The R side of the polyglot contract. Reads one JSON request on stdin, writes
# one JSON response on stdout, and puts nothing else on stdout ever - messages
# go to stderr, because a stray cat() would corrupt the response.
#
# Survival modelling stays in R deliberately. Reimplementing `survival` in
# Python would cost weeks and be worse than the original; integrating cleanly
# across a declared contract is the stronger engineering signal.
#
# Request:
#   {"action": "cox_persd" | "cox_multivariable" | "quartile",
#    "time":   [numeric, months],
#    "event":  [0/1],
#    "scores": {"name": [numeric], ...},
#    "covariates": {"name": [numeric], ...}   # optional
#   }
#
# Response:
#   {"ok": true, "action": ..., "r_version": ..., "results": [...]}
#   {"ok": false, "error": "..."}
##############################################################################

suppressWarnings(suppressMessages({
  library(survival)
  library(jsonlite)
}))

fail <- function(msg) {
  # `ok` must be a JSON scalar, not a length-1 array: a client that reads
  # [false] as truthy would treat a refused request as a successful one.
  cat(toJSON(list(ok = unbox(FALSE), error = unbox(msg)), auto_unbox = FALSE), "\n", sep = "")
  quit(status = 1)
}

# ---------------------------------------------------------------- validation
validate <- function(req) {
  for (field in c("action", "time", "event")) {
    if (is.null(req[[field]])) fail(paste0("request is missing required field: ", field))
  }
  n <- length(req$time)
  if (length(req$event) != n) {
    fail(sprintf("time and event lengths differ (%d vs %d)", n, length(req$event)))
  }
  if (n < 10) fail(sprintf("only %d observations; refusing to fit", n))
  if (!all(req$event %in% c(0, 1))) fail("event must be coded 0/1")
  if (any(!is.finite(req$time)) || any(req$time <= 0)) {
    fail("time must be finite and strictly positive")
  }
  for (nm in names(req$scores)) {
    if (length(req$scores[[nm]]) != n) {
      fail(sprintf("score '%s' has length %d, expected %d", nm, length(req$scores[[nm]]), n))
    }
  }
  invisible(TRUE)
}

# ------------------------------------------------------------------ helpers
cox_row <- function(fit, term, label, model, n, n_events, extra = list()) {
  s <- summary(fit)
  ci <- s$conf.int
  co <- s$coefficients
  c(
    list(
      name       = unbox(label),
      model      = unbox(model),
      term       = unbox(term),
      n          = unbox(n),
      n_events   = unbox(n_events),
      hr         = unbox(unname(co[term, "exp(coef)"])),
      ci_low     = unbox(unname(ci[term, "lower .95"])),
      ci_high    = unbox(unname(ci[term, "upper .95"])),
      p          = unbox(unname(co[term, "Pr(>|z|)"])),
      c_index    = unbox(unname(s$concordance["C"])),
      c_index_se = unbox(unname(s$concordance["se(C)"]))
    ),
    extra
  )
}

# ------------------------------------------------------------------- actions
run_cox_persd <- function(req) {
  # Per-SD hazard ratio: the score is standardised so that the HR is per one
  # standard deviation, which is what makes signatures of different scales
  # comparable to each other.
  out <- list()
  for (nm in names(req$scores)) {
    z <- as.numeric(scale(as.numeric(req$scores[[nm]])))
    ok <- is.finite(z) & is.finite(req$time) & is.finite(req$event)
    d <- data.frame(time = req$time[ok], event = req$event[ok], z = z[ok])
    if (length(unique(d$z)) < 5) {
      out[[length(out) + 1]] <- list(
        name = unbox(nm), model = unbox("per_sd"), error = unbox("score has <5 distinct values")
      )
      next
    }
    fit <- coxph(Surv(time, event) ~ z, data = d)
    out[[length(out) + 1]] <- cox_row(fit, "z", nm, "per_sd", nrow(d), sum(d$event))
  }
  out
}

run_cox_multivariable <- function(req) {
  # Every score plus any covariates in one model: does a signature add
  # information beyond what is already on the table?
  d <- data.frame(time = req$time, event = req$event)
  terms <- character(0)
  for (nm in names(req$scores)) {
    d[[nm]] <- as.numeric(scale(as.numeric(req$scores[[nm]])))
    terms <- c(terms, nm)
  }
  for (nm in names(req$covariates)) {
    d[[nm]] <- as.numeric(scale(as.numeric(req$covariates[[nm]])))
    terms <- c(terms, nm)
  }
  d <- d[complete.cases(d), ]
  if (nrow(d) < 10) fail("fewer than 10 complete cases for the multivariable model")
  if (sum(d$event) < length(terms) * 5) {
    message(sprintf(
      "warning: %d events for %d terms - fewer than 5 events per variable",
      sum(d$event), length(terms)
    ))
  }
  form <- as.formula(paste("Surv(time, event) ~", paste(terms, collapse = " + ")))
  fit <- coxph(form, data = d)
  lapply(terms, function(t) {
    cox_row(fit, t, t, "multivariable", nrow(d), sum(d$event),
            extra = list(terms = paste(terms, collapse = "+")))
  })
}

run_quartile <- function(req) {
  # Top vs bottom quartile. Reported alongside per-SD, never instead of it:
  # with few events the quartile estimate has intervals too wide to interpret.
  out <- list()
  for (nm in names(req$scores)) {
    s <- as.numeric(req$scores[[nm]])
    q <- quantile(s, probs = c(0.25, 0.75), na.rm = TRUE)
    grp <- ifelse(s <= q[1], "Q1", ifelse(s >= q[2], "Q4", NA))
    keep <- !is.na(grp)
    d <- data.frame(
      time = req$time[keep], event = req$event[keep],
      grp = factor(grp[keep], levels = c("Q1", "Q4"))
    )
    if (nlevels(droplevels(d$grp)) < 2) {
      out[[length(out) + 1]] <- list(
        name = unbox(nm), model = unbox("quartile"), error = unbox("only one quartile group")
      )
      next
    }
    fit <- coxph(Surv(time, event) ~ grp, data = d)
    row <- cox_row(fit, "grpQ4", nm, "quartile", nrow(d), sum(d$event),
                   extra = list(
                     n_q1 = unbox(sum(d$grp == "Q1")),
                     n_q4 = unbox(sum(d$grp == "Q4")),
                     events_q1 = unbox(sum(d$event[d$grp == "Q1"])),
                     events_q4 = unbox(sum(d$event[d$grp == "Q4"]))
                   ))
    out[[length(out) + 1]] <- row
  }
  out
}

# ---------------------------------------------------------------------- main
main <- function() {
  raw <- paste(readLines(file("stdin"), warn = FALSE), collapse = "\n")
  if (!nzchar(trimws(raw))) fail("empty request on stdin")
  req <- tryCatch(fromJSON(raw, simplifyVector = TRUE),
                  error = function(e) fail(paste("malformed JSON request:", conditionMessage(e))))
  validate(req)

  results <- switch(req$action,
    cox_persd         = run_cox_persd(req),
    cox_multivariable = run_cox_multivariable(req),
    quartile          = run_quartile(req),
    fail(paste0("unknown action: ", req$action))
  )

  cat(toJSON(list(
    ok        = unbox(TRUE),
    action    = unbox(req$action),
    r_version = unbox(paste(R.version$major, R.version$minor, sep = ".")),
    survival_version = unbox(as.character(packageVersion("survival"))),
    results   = results
  ), auto_unbox = FALSE, digits = 12, na = "null"), "\n", sep = "")
}

main()
