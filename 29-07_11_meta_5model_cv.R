# 29-07_11_meta_5model_cv.R
#
# REQUIRES: oof_clogit_ranger_xgb.csv, oof_two_stage.csv, oof_glmnet.csv on disk
# (same 3 files 29-07_10 reads). mlogloss() must be defined. Standalone otherwise —
# does its own train/fold load, doesn't need 29-07_2's session state.
#
# Purpose: CV-check whether dropping lc13 (Section 1 of 29-07_10, gmnl full-data
# fit broken) meaningfully changes Model 31's CV. Same fold-loop structure as
# 29-07_8, just re-run with feat_cols minus the 4 lc13_Ch* columns (20 vs 24 cols).
# Not a new model number — a feature-set variant of Model 31, checked before
# submitting the 5-model version in 29-07_10.

library(data.table); library(xgboost)

train <- fread("train.csv")
set.seed(2024)
case_ids <- unique(train$Case)
case_fold <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train <- merge(train, case_fold, by = "Case")

oof_base <- fread("oof_clogit_ranger_xgb.csv")
oof_ts   <- fread("oof_two_stage.csv")
oof_gl   <- fread("oof_glmnet.csv")

oof_all <- merge(oof_base, oof_ts, by = c("Case", "Task"))
oof_all <- merge(oof_all, oof_gl, by = c("Case", "Task"))

fold_map <- unique(train[, .(Case, fold)])
oof_all  <- merge(oof_all, fold_map, by = "Case")
stopifnot(nrow(oof_all) == nrow(train))

feat_cols <- c("clogit15_Ch1","clogit15_Ch2","clogit15_Ch3","clogit15_Ch4",
               "ranger18_Ch1","ranger18_Ch2","ranger18_Ch3","ranger18_Ch4",
               "xgb20_Ch1","xgb20_Ch2","xgb20_Ch3","xgb20_Ch4",
               "ts_Ch1","ts_Ch2","ts_Ch3","ts_Ch4",
               "glmnet30_Ch1","glmnet30_Ch2","glmnet30_Ch3","glmnet30_Ch4")

y_class0 <- max.col(oof_all[, .(Ch1, Ch2, Ch3, Ch4)]) - 1
meta_oof <- matrix(NA_real_, nrow(oof_all), 4)

# same params/nrounds as the logged 6-model Model 31 (29-07_8/9) - unchanged,
# so any CV difference is attributable to dropping lc13, not a param change
params <- list(objective = "multi:softprob", num_class = 4,
               eta = 0.05, max_depth = 2, subsample = 0.8, colsample_bytree = 0.8)
nrounds <- 150

for (f in sort(unique(oof_all$fold))) {
  tr <- oof_all$fold != f
  te <- oof_all$fold == f

  dtr <- xgb.DMatrix(as.matrix(oof_all[tr, ..feat_cols]), label = y_class0[tr])
  dte <- xgb.DMatrix(as.matrix(oof_all[te, ..feat_cols]))

  fit  <- xgb.train(params, dtr, nrounds = nrounds)
  pred <- predict(fit, dte)

  if (f == sort(unique(oof_all$fold))[1]) {
    cat("predict() class:", class(pred), " dim:", paste(dim(pred), collapse = "x"), "\n")
  }
  if (!is.null(dim(pred))) {
    meta_oof[te, ] <- pred
  } else {
    meta_oof[te, ] <- matrix(pred, ncol = 4, byrow = TRUE)
  }
}

actual_mat <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
meta_cv_5model <- mlogloss(actual_mat, meta_oof)
cat("\n5-model meta-learner (no lc13) CV:", meta_cv_5model, "\n")
cat("vs 6-model Model 31 (logged):        1.152278 (seed 2024) / 1.152751 (seed 99)\n")
cat("vs Model 29 (best linear ensemble):  1.154955\n")
