# 29-07_7_gmnl_full_cv.R
# REQUIRES (run first, in order): rm(list=ls()), then
#   27-07_1_pipeline_foundation.R
#   27-07_10_clogit_simplified_segment.R
# (need train_long, fold, rhs_vars, mlogloss(), train). Needs `gmnl` package.
#
# Continues Next Steps #23 (gmnl mixed-logit, deferred from 29-07_5 diagnostic,
# which PASSED on a 200-case slice: converged, Price mean -2.400486, sd.Price
# 2.504052). Formula matches that diagnostic: rhs_vars only (no demo
# interactions) with a random (normal) Price coefficient.
#
# STEP A first (single fold, ~1/5 the cost): confirms predict.gmnl() actually
# works and shows its output shape -- untested in this environment so far.
# Do NOT run Step B until Step A's output is pasted back and reviewed.

library(gmnl)
library(mlogit)
library(data.table)

# ---- formula (same spec as 29-07_5 diagnostic) ----
gmnl_formula <- as.formula(paste0(
  "chosen ~ ", paste(rhs_vars, collapse = " + "), " | 0"
))

# ---- mlogit.data prep (id.var="Case" needed for panel=TRUE, per 29-07_5 fix) ----
gmnl_data_prep <- function(dt) {
  d <- copy(dt)
  d[, chosen := as.logical(chosen)]
  mlogit.data(
    as.data.frame(d),
    choice  = "chosen",
    shape   = "long",
    chid.var = "chid",
    alt.var  = "alt",
    id.var   = "Case"
  )
}

R_DRAWS <- 50  # diagnostic didn't log its R value -- start modest, can raise if fast

# =============================================================================
# STEP A-FAST -- mechanics-only check. Minimum data/draws/iterations needed to
# answer ONE question: does predict.gmnl(newdata=...) work, and what shape does
# it return? NOT trying to converge a real model here (29-07_5 already proved
# convergence works on a 200-case slice) -- this only derisks predict(), which
# is the untested part. Should run in well under a minute.
# Run this FIRST. Only proceed to full-fold Step A below if this succeeds AND
# you still have time-box budget left.
# =============================================================================
set.seed(42)
all_cases   <- unique(train_long$Case)
mech_cases  <- sample(all_cases, 300)
val_cases   <- sample(setdiff(all_cases, mech_cases), 100)
mech_tr <- train_long[Case %in% mech_cases]
mech_va <- train_long[Case %in% val_cases]

t0 <- Sys.time()
fit_mech <- gmnl(gmnl_formula, data = gmnl_data_prep(mech_tr), model = "mixl",
                 ranp = c(Price = "n"), R = 10, panel = TRUE, method = "bfgs",
                 iterlim = 50)
cat("Mechanics fit time:", as.numeric(Sys.time() - t0, units = "secs"), "sec\n")
cat("Mechanics fit iterations used:", fit_mech$logLik$iterations, "/ 50\n")
cat("(convergence not the goal here -- fine either way)\n")

t1 <- Sys.time()
pred_mech <- predict(fit_mech, newdata = gmnl_data_prep(mech_va))
cat("Mechanics predict time:", as.numeric(Sys.time() - t1, units = "secs"), "sec\n")

# This is the actual thing we need answered:
cat("class(pred_mech):", class(pred_mech), "\n")
print(dim(pred_mech))
print(head(pred_mech))
print(colnames(pred_mech))
print(rownames(pred_mech)[1:5])

# ---- STOP HERE and paste back the block above before running anything below ----

# =============================================================================
# STEP A -- single fold (fold 1), timed, inspect predict() output before looping.
# Only run after Step A-fast confirms predict() works and you have time budget.
# =============================================================================
tr_1 <- train_long[fold != 1]
va_1 <- copy(train_long[fold == 1])

ITERLIM <- 500  # hard iteration cap -- prevents an unconverged fit from running indefinitely.
# NOT a substitute for actual convergence: if the cap is hit, treat the fit as
# "unscored" (mechanics-check only), not a real CV number -- see convergence
# check immediately below.

t0 <- Sys.time()
fit_1 <- gmnl(gmnl_formula, data = gmnl_data_prep(tr_1), model = "mixl",
              ranp = c(Price = "n"), R = R_DRAWS, panel = TRUE, method = "bfgs",
              iterlim = ITERLIM)
cat("Fold 1 fit time:", as.numeric(Sys.time() - t0, units = "mins"), "min\n")

# ---- explicit convergence check -- do NOT trust logLik/coef below if this says otherwise ----
conv_code <- fit_1$logLik$code       # maxLik convergence code (1-2 = genuine convergence; varies by method)
conv_msg  <- fit_1$logLik$message
n_iter    <- fit_1$logLik$iterations
cat("Fold 1 convergence code:", conv_code, "-", conv_msg, "\n")
cat("Fold 1 iterations used:", n_iter, "/ cap", ITERLIM, "\n")
if (!is.null(n_iter) && n_iter >= ITERLIM) {
  cat("*** HIT ITERATION CAP -- likely did NOT converge. Treat as mechanics-check only,\n")
  cat("*** do not log the log-loss/coefficients below as a real result. ***\n")
}
cat("Fold 1 logLik (informational only if not converged):", as.numeric(fit_1$logLik$maximum), "\n")

t1 <- Sys.time()
pred_1 <- predict(fit_1, newdata = gmnl_data_prep(va_1))
cat("Fold 1 predict time:", as.numeric(Sys.time() - t1, units = "mins"), "min\n")

# Inspect -- paste this output back before running Step B
cat("class(pred_1):", class(pred_1), "\n")
print(dim(pred_1))
print(head(pred_1))
print(colnames(pred_1))

# =============================================================================
# STEP B -- full 5-fold CV. DO NOT RUN until Step A output confirms predict()
# returns a chid x 4 probability matrix (adjust the alignment block below if
# the column names/order differ from what's printed above).
# =============================================================================
# score_gmnl_cv <- function(data_long, actual_full, R_draws = R_DRAWS) {
#   fold_loss <- numeric(5)
#   for (i in 1:5) {
#     tr_i <- data_long[fold != i]
#     va_i <- copy(data_long[fold == i])
#
#     fit <- tryCatch(
#       gmnl(gmnl_formula, data = gmnl_data_prep(tr_i), model = "mixl",
#            ranp = c(Price = "n"), R = R_draws, panel = TRUE, method = "bfgs"),
#       error = function(e) { cat("Fold", i, "fit failed:", conditionMessage(e), "\n"); NULL }
#     )
#     if (is.null(fit)) { fold_loss[i] <- NA; next }
#
#     pred <- tryCatch(
#       predict(fit, newdata = gmnl_data_prep(va_i)),
#       error = function(e) { cat("Fold", i, "predict failed:", conditionMessage(e), "\n"); NULL }
#     )
#     if (is.null(pred)) { fold_loss[i] <- NA; next }
#
#     # ADAPT to Step A's actual output shape before uncommenting/running:
#     # need a chid-ordered n x 4 matrix aligned to Ch1..Ch4 for mlogloss().
#     va_wide_order <- unique(va_i[, .(chid, Case, Task)])
#     setkey(va_wide_order, chid)
#     pred_dt <- as.data.table(pred)
#     pred_dt[, chid := rownames(pred)]
#     preds_wide <- merge(va_wide_order, pred_dt, by = "chid", sort = FALSE)
#
#     actual_aligned <- merge(preds_wide[, .(Case, Task)], actual_full, by = c("Case","Task"), sort = FALSE)
#     actual   <- as.matrix(actual_aligned[, .(Ch1, Ch2, Ch3, Ch4)])
#     pred_mat <- as.matrix(preds_wide[, .(Ch1, Ch2, Ch3, Ch4)])  # colnames may need renaming first
#
#     fold_loss[i] <- mlogloss(actual, pred_mat)
#     cat("Fold", i, "gmnl log-loss:", fold_loss[i], "\n")
#   }
#   fold_loss
# }
#
# actual_lookup <- train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]
# gmnl_scores <- score_gmnl_cv(train_long, actual_lookup)
# cat("\ngmnl mixed logit -- CV mean:", mean(gmnl_scores, na.rm=TRUE),
#     "sd:", sd(gmnl_scores, na.rm=TRUE), "\n")