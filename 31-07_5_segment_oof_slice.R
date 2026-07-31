# 31-07_5_segment_oof_slice.R
# Purpose: per-segment CV logloss slice on Model 38's OOF predictions. Diagnostic only, no new model.
# PREREQS: run 27-07_1 and 27-07_4 manually first (once per session). Then source() the rest below,
# in order, via the run_files loop. Needs: d_train_ext2, feat_cols_ext2, fold_vec, train (segment/Ch1-4).

library(xgboost)
library(data.table)

stopifnot(exists("d_train_ext2"), exists("feat_cols_ext2"), exists("fold_vec"), exists("train"))

# --- Reproduce Model 38's OOF probs (xgb.cv w/ prediction=TRUE; plain xgb.cv never saved these) ---
# VERIFY: label source. d_train_ext2 should carry Ch1-4 (from oof_all) - derive integer label from these,
# not a "y" column (unconfirmed whether one exists on d_train_ext2).
y_ext <- max.col(d_train_ext2[, .(Ch1, Ch2, Ch3, Ch4)]) - 1L

dtrain_ext2 <- xgb.DMatrix(
  data  = as.matrix(d_train_ext2[, ..feat_cols_ext2]),
  label = y_ext
)

folds_list <- lapply(1:5, function(k) which(fold_vec == k))

params_model38 <- list(
  objective   = "multi:softprob",
  num_class   = 4,
  eval_metric = "mlogloss",
  max_depth   = 3,
  eta         = 0.02
)

cv_model38 <- xgb.cv(
  params    = params_model38,
  data      = dtrain_ext2,
  nrounds   = 286,
  folds     = folds_list,
  prediction = TRUE,
  verbose   = 0
)

# reproduce known-good CV before trusting the slice (project rule)
cv_logloss <- min(cv_model38$evaluation_log$test_mlogloss_mean)
cat("Reproduced CV logloss:", cv_logloss, "(expect ~1.148969)\n")
stopifnot(abs(cv_logloss - 1.148969) < 0.001)

oof_preds <- cv_model38$cv_predict$pred  # Gotcha #30: not $pred
colnames(oof_preds) <- paste0("p", 1:4)

# --- Attach segment + actuals, keyed on Case+Task ---
seg_lookup <- unique(train[, .(Case, Task, segment)])

oof_dt <- data.table(Case = d_train_ext2$Case, Task = d_train_ext2$Task, oof_preds)
oof_dt <- merge(oof_dt, seg_lookup, by = c("Case", "Task"), all.x = TRUE)
oof_dt <- merge(oof_dt, train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)], by = c("Case", "Task"), all.x = TRUE)
stopifnot(!anyNA(oof_dt$segment))

# --- Per-row, then per-segment logloss ---
eps <- 1e-15
oof_dt[, ll := -(Ch1 * log(pmax(p1, eps)) + Ch2 * log(pmax(p2, eps)) +
                  Ch3 * log(pmax(p3, eps)) + Ch4 * log(pmax(p4, eps)))]

seg_summary <- oof_dt[, .(n = .N, mean_logloss = mean(ll)), by = segment][order(-mean_logloss)]
print(seg_summary)
