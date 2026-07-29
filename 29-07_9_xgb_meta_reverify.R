# Requires: oof_clogit_ranger_xgb.csv (28-07_1), oof_latent_class.csv (28-07_2),
#           oof_two_stage.csv (28-07_12), oof_glmnet.csv (29-07_4) — all already generated.
# Fresh-seed reverification of 29-07_8 (Model 31 candidate, CV 1.152278).
# Only the fold-split seed changes (2024 -> 99); model params/structure identical,
# so this isolates fold-split sensitivity, not a retune.
# mlogloss() must already be defined (sourced earlier in session).

library(data.table)
library(xgboost)

train <- fread("train.csv")

set.seed(99)
case_ids <- unique(train$Case)
case_fold <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train <- merge(train, case_fold, by = "Case")

oof_base <- fread("oof_clogit_ranger_xgb.csv")
oof_lc   <- fread("oof_latent_class.csv")
oof_ts   <- fread("oof_two_stage.csv")
oof_gl   <- fread("oof_glmnet.csv")

oof_all <- merge(oof_base, oof_lc, by = c("Case", "Task"))
oof_all <- merge(oof_all, oof_ts, by = c("Case", "Task"))
oof_all <- merge(oof_all, oof_gl, by = c("Case", "Task"))

fold_map <- unique(train[, .(Case, fold)])
oof_all  <- merge(oof_all, fold_map, by = "Case")

feat_cols <- c("clogit15_Ch1","clogit15_Ch2","clogit15_Ch3","clogit15_Ch4",
               "ranger18_Ch1","ranger18_Ch2","ranger18_Ch3","ranger18_Ch4",
               "xgb20_Ch1","xgb20_Ch2","xgb20_Ch3","xgb20_Ch4",
               "lc13_Ch1","lc13_Ch2","lc13_Ch3","lc13_Ch4",
               "ts_Ch1","ts_Ch2","ts_Ch3","ts_Ch4",
               "glmnet30_Ch1","glmnet30_Ch2","glmnet30_Ch3","glmnet30_Ch4")

y_class0 <- max.col(oof_all[, .(Ch1, Ch2, Ch3, Ch4)]) - 1
meta_oof <- matrix(NA_real_, nrow(oof_all), 4)

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
  meta_oof[te, ] <- pred   # confirmed matrix-shaped in this env (29-07_8 check)
}

actual_mat <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
meta_cv_reverify <- mlogloss(actual_mat, meta_oof)

cat("meta_cv_reverify (seed 99):", meta_cv_reverify, "\n")
cat("original (seed 2024):     ", 1.152278, "\n")
