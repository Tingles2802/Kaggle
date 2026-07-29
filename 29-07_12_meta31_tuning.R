# 29-07_12_meta31_tuning.R
#
# REQUIRES: oof_clogit_ranger_xgb.csv, oof_two_stage.csv, oof_glmnet.csv on disk.
# mlogloss() must be defined. Standalone — does its own train/fold load, same as 29-07_11.
#
# Purpose: grid search Model 31's (5-model) xgboost meta-learner hyperparams.
# Current submitted config (max_depth=2, eta=0.05, nrounds=150) was deliberately
# shallow/untuned (CV 1.152292, LB 1.200). Grid per Next Steps #29.

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

y_class0   <- max.col(oof_all[, .(Ch1, Ch2, Ch3, Ch4)]) - 1
actual_mat <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
folds      <- sort(unique(oof_all$fold))

grid <- CJ(max_depth = c(2, 3, 4),
           eta       = c(0.03, 0.05, 0.1),
           nrounds   = c(100, 150, 200, 300))
grid[, cv_logloss := NA_real_]

for (g in seq_len(nrow(grid))) {
  params <- list(objective = "multi:softprob", num_class = 4,
                  eta = grid$eta[g], max_depth = grid$max_depth[g],
                  subsample = 0.8, colsample_bytree = 0.8)
  meta_oof <- matrix(NA_real_, nrow(oof_all), 4)

  for (f in folds) {
    tr <- oof_all$fold != f
    te <- oof_all$fold == f
    dtr <- xgb.DMatrix(as.matrix(oof_all[tr, ..feat_cols]), label = y_class0[tr])
    dte <- xgb.DMatrix(as.matrix(oof_all[te, ..feat_cols]))
    fit  <- xgb.train(params, dtr, nrounds = grid$nrounds[g])
    pred <- predict(fit, dte)
    meta_oof[te, ] <- if (!is.null(dim(pred))) pred else matrix(pred, ncol = 4, byrow = TRUE)
  }

  grid$cv_logloss[g] <- mlogloss(actual_mat, meta_oof)
  cat(sprintf("depth=%d eta=%.2f nrounds=%d -> CV %.6f\n",
              grid$max_depth[g], grid$eta[g], grid$nrounds[g], grid$cv_logloss[g]))
}

setorder(grid, cv_logloss)
cat("\nTop 5 configs:\n"); print(head(grid, 5))
cat("\nCurrent submitted config (depth=2, eta=0.05, nrounds=150) CV: 1.152292\n")
