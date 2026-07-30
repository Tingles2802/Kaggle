# 29-07_20_submission_model32.R
#
# REQUIRES: run 29-07_2_submission_model29.R, then 29-07_10_submission_model31.R
# FIRST, in the same session, unmodified. Reuses without recomputation:
#   oof_all, feat_cols, y_class0_full, d_meta_train, preds_all6, d_meta_test
#
# Purpose: refit ONLY the meta-learner with Model 32's tuned hyperparams
# (depth=3, eta=0.03, nrounds=200 vs Model 31's depth=2/eta=0.05/nrounds=150),
# same 5-model/20-col feat_cols set and same full training OOF stack as 29-07_10.
# Base models (clogit/ranger/xgb/two-stage/glmnet) are NOT refit here — reuses
# 29-07_10's preds_all6 test-set base predictions as-is.
# Writes submission_29-07_2.csv.

library(xgboost); library(data.table)

stopifnot(exists("d_meta_train"), exists("d_meta_test"), exists("preds_all6"), exists("feat_cols"))

# --- Model 32 tuned hyperparams (29-07_12 grid winner, fresh-seed reverified 29-07_17) ---
params_tuned <- list(objective = "multi:softprob", num_class = 4,
                      eta = 0.03, max_depth = 3, subsample = 0.8, colsample_bytree = 0.8)
nrounds_tuned <- 200

set.seed(2024)
fit_meta32_full <- xgb.train(params_tuned, d_meta_train, nrounds = nrounds_tuned)  # final refit, no held-out fold

meta_pred32 <- predict(fit_meta32_full, d_meta_test)
cat("predict() class:", class(meta_pred32), " dim:", paste(dim(meta_pred32), collapse = "x"), "\n")
if (is.null(dim(meta_pred32))) meta_pred32 <- matrix(meta_pred32, ncol = 4, byrow = TRUE)  # gotcha #19 guard
colnames(meta_pred32) <- c("Ch1","Ch2","Ch3","Ch4")

cat("Row-sum range pre-normalize (should already be ~1):", range(rowSums(meta_pred32)), "\n")
meta_pred32 <- meta_pred32 / rowSums(meta_pred32)

stopifnot(!any(is.na(meta_pred32)))
stopifnot(all(meta_pred32 >= -1e-8 & meta_pred32 <= 1 + 1e-8))

submission32 <- data.table(No = preds_all6$No, meta_pred32)
setorder(submission32, No)

stopifnot(nrow(submission32) == 4997)
stopifnot(identical(names(submission32), c("No","Ch1","Ch2","Ch3","Ch4")))
stopifnot(all(abs(rowSums(submission32[, .(Ch1,Ch2,Ch3,Ch4)]) - 1) < 1e-8))

fwrite(submission32, "submission_30-07_1.csv")
cat("\nSaved submission_30-07_1.csv (", nrow(submission32), "rows )\n")
print(head(submission32))
