# 29-07_3_glmnet_multinomial.R
# REQUIRES (run first, in order): 27-07_1_pipeline_foundation.R,
#   27-07_4_trees_setup.R (Steps 1-3, for ranger_data + cat_feature_cols)
#
# Purpose: CV-score a regularized multinomial logit (glmnet, ridge) as a new,
# structurally different base model (individual-level, all-attributes-at-once,
# regularized) for ensemble diversity. Uses the same case-grouped 5-fold split
# as every other model (train$fold) for direct comparability.
#
# NOTE: demo cols (segmentind/genderind/educind/regionind/etc.) are fed as raw
# integer codes, same convention as 27-07_4's trees — imposes an ordinal
# assumption that's harmless for trees but not strictly correct for a linear
# model. Kept as-is for this first pass; revisit (factor-encode) only if this
# score looks promising enough to invest further.
#
# This is a CV-scoring pass only — no OOF-for-ensemble generation yet. That
# follows separately (new script, same pattern as 28-07_1) only if this beats
# a sanity bar worth ensembling.

library(glmnet)
library(data.table)

# ---- 1. Design matrix ----
# Reuse ranger_data (19 attrs x4 as factors, Price/demo numeric, has y+fold).
# Alt 4 is a structural opt-out (see experiment_log Section 2.1): all 19
# attribute cols for alt 4 are fixed at 0 for every row, so as factors they
# have a single level — model.matrix's contrasts<- can't handle that and
# errors. Detect + drop constant columns generically (also catches Price4,
# constant at 0, which wouldn't itself error but carries no information).
raw_X <- ranger_data[, !c("y", "fold"), with = FALSE]
is_constant <- vapply(raw_X, function(col) length(unique(col)) <= 1, logical(1))
if (any(is_constant)) {
  cat("Dropping constant columns (structural alt-4 opt-out):\n")
  print(names(raw_X)[is_constant])
  raw_X <- raw_X[, !names(raw_X)[is_constant], with = FALSE]
}

# Full dummy encoding (no intercept) so every remaining attribute level gets
# its own coefficient — glmnet's L2 penalty handles the resulting collinearity.
X_full <- model.matrix(~ . - 1, data = raw_X)
y_full <- ranger_data$y      # factor, levels "1".."4" — must match Ch1..Ch4 order
fold_vec <- ranger_data$fold

stopifnot(nrow(X_full) == nrow(train))
stopifnot(identical(levels(y_full), c("1","2","3","4")))

# ---- 2. Manual 5-fold CV ----
# Outer fold = same case-grouped fold as every other model.
# Inner CV (cv.glmnet's own, on outer-training rows only) picks lambda —
# no leakage into the held-out outer fold.
cv_scores_glmnet <- numeric(5)
oof_preds <- matrix(NA_real_, nrow = nrow(train), ncol = 4)

for (k in 1:5) {
  cat("=== Fold", k, "===\n")
  tr_idx <- which(fold_vec != k)
  va_idx <- which(fold_vec == k)
  
  fit_cv <- cv.glmnet(
    X_full[tr_idx, ], y_full[tr_idx],
    family = "multinomial", type.multinomial = "grouped",
    alpha = 0, nfolds = 5, standardize = TRUE
  )
  
  preds <- predict(fit_cv, newx = X_full[va_idx, ], s = "lambda.min", type = "response")
  preds <- preds[, , 1]  # drop glmnet's redundant 3rd array dim
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  
  actual <- as.matrix(train[va_idx, .(Ch1, Ch2, Ch3, Ch4)])
  cv_scores_glmnet[k] <- mlogloss(actual, preds)
  oof_preds[va_idx, ] <- preds
  cat("glmnet fold", k, "log-loss:", cv_scores_glmnet[k], " lambda.min:", fit_cv$lambda.min, "\n")
}

cat("\nglmnet (ridge) CV — Mean:", mean(cv_scores_glmnet), " SD:", sd(cv_scores_glmnet), "\n")
cat("vs current best standalone (Model 15, clogit): 1.162469\n")
cat("vs current best ensemble (Model 29): 1.154955\n")

# Hard sanity check — worse than uniform benchmark means a pipeline bug, not
# a real result (see experiment_log Gotcha list). Stop and debug, don't tune.
stopifnot(mean(cv_scores_glmnet) < 1.386294)