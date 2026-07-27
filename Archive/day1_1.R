# =============================================================
# Day 1 — CV harness + log-loss function + baseline models
# Safety Bundle Choice Competition
# =============================================================
# Run this in your own R environment (RStudio / Kaggle notebook).
# Paste back: (a) any errors, (b) the printed log-loss numbers,
# (c) whether submission_day1_uniform.csv uploaded cleanly to Kaggle
#     and what public LB score it got.
# -------------------------------------------------------------
# 
library(data.table)

set.seed(2024)

train <- fread("train.csv")
test  <- fread("test.csv")

# ---------------------------------------------------------------
# 1. Log-loss function — matches competition formula exactly
#    LogLoss = -(1/n) * sum_i sum_j y_ij * log(p_ij)
#    Clipping avoids log(0) blowing up on overconfident wrong preds.
# ---------------------------------------------------------------
mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmin(pmax(pred_mat, eps), 1 - eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

# ---------------------------------------------------------------
# 2. Case-grouped K-fold CV
#    WHY grouped by Case and not by row: test.csv respondents (Case)
#    do NOT overlap with train.csv respondents (confirmed: 0 overlap,
#    1135 train cases vs 263 test cases, each with 19 tasks). A random
#    row-level split would let 18 of a respondent's other tasks leak
#    into the training fold, making CV optimistic vs. the real test
#    scenario (entirely unseen people). Every task for a given Case
#    must land in the same fold.
# ---------------------------------------------------------------
make_case_folds <- function(cases, k = 5, seed = 2024) {
  set.seed(seed)
  uniq_cases  <- unique(cases)
  fold_lookup <- sample(rep(1:k, length.out = length(uniq_cases)))
  names(fold_lookup) <- uniq_cases
  fold_lookup[as.character(cases)]
}

train[, fold := make_case_folds(Case, k = 5, seed = 2024)]

actual_mat <- as.matrix(train[, .(Ch1, Ch2, Ch3, Ch4)])

# ---------------------------------------------------------------
# 3. Baseline 1: Uniform (0.25 each) — pipeline sanity check
# ---------------------------------------------------------------
uniform_pred <- matrix(0.25, nrow = nrow(train), ncol = 4)
cat("Uniform baseline log-loss:", mlogloss(actual_mat, uniform_pred), "\n")
# Should print 1.38629 (matches the stated competition benchmark).
# If it doesn't, something in the log-loss function or data load is off.

# ---------------------------------------------------------------
# 4. Baseline 2: Marginal frequency (overall alternative share)
#    In-sample version (for reference only — optimistic, not a CV score)
# ---------------------------------------------------------------
marg_freq <- colMeans(actual_mat)
cat("Overall alternative shares (Ch1..Ch4):", round(marg_freq, 4), "\n")

marg_pred_insample <- matrix(marg_freq, nrow = nrow(train), ncol = 4, byrow = TRUE)
cat("Marginal-frequency in-sample log-loss:",
    mlogloss(actual_mat, marg_pred_insample), "\n")

# ---------------------------------------------------------------
# 5. Baseline 2, proper out-of-fold CV estimate
#    (frequency computed on training folds only, applied to held-out fold)
# ---------------------------------------------------------------
cv_marg_loss <- numeric(5)
for (f in 1:5) {
  tr <- train[fold != f]
  va <- train[fold == f]
  freq <- colMeans(as.matrix(tr[, .(Ch1, Ch2, Ch3, Ch4)]))
  pred <- matrix(freq, nrow = nrow(va), ncol = 4, byrow = TRUE)
  cv_marg_loss[f] <- mlogloss(as.matrix(va[, .(Ch1, Ch2, Ch3, Ch4)]), pred)
}
cat("CV marginal-frequency log-loss — mean:", round(mean(cv_marg_loss), 5),
    " sd:", round(sd(cv_marg_loss), 5), "\n")
cat("Per-fold scores:", round(cv_marg_loss, 5), "\n")

# ---------------------------------------------------------------
# 6. Day 1 baseline submission (uniform) — format/pipeline check only
#    Confirms our submission format matches sample_submission.csv and
#    that CV log-loss (1.38629) matches what Kaggle reports on the
#    public leaderboard for this trivial baseline.
# ---------------------------------------------------------------
submission <- data.table(
  No  = test$No,
  # Ch1 = 0.25, Ch2 = 0.25, Ch3 = 0.25, Ch4 = 0.25
)
fwrite(submission, "submission_day1_uniform.csv")
cat("Wrote submission_day1_uniform.csv —", nrow(submission), "rows\n")

# =============================================================
# Paste back to Claude:
#   - the 5 printed numbers/lines above
#   - confirmation the uniform baseline log-loss == 1.38629
#   - (if you submit it) the public LB score, to confirm CV/LB alignment
# =============================================================


table(train$fold)                       # are folds populated 1-5, roughly even counts?
str(train[, .(Ch1, Ch2, Ch3, Ch4)])      # confirms these are still integer 0/1, not something coerced

