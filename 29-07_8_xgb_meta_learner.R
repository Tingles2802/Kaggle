# Requires: oof_clogit_ranger_xgb.csv (28-07_1), oof_latent_class.csv (28-07_2),
#           oof_two_stage.csv (28-07_12), oof_glmnet.csv (29-07_4) — all already generated.
# train.csv loaded fresh; fold assignment self-generated (grouped by Case, seed 2024),
# same convention as 28-07_8 Model 23 — not required to match original modeling folds,
# only needs to be a valid grouped split. Kept identical to 28-07_8 for direct comparison.
# mlogloss() must already be defined (sourced earlier in session).

library(data.table)
library(xgboost)

train <- fread("train.csv")

set.seed(2024)
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

# xgboost multiclass labels must be 0-indexed
y_class0 <- max.col(oof_all[, .(Ch1, Ch2, Ch3, Ch4)]) - 1
meta_oof <- matrix(NA_real_, nrow(oof_all), 4)

# Shallow/low-eta on purpose: 24 features are highly correlated probability
# outputs from 6 models already fit on the same data — deep trees/many rounds
# risk overfitting the stack itself. No early stopping (avoids optimism from
# tuning rounds against the same fold being scored). Revisit only if this
# looks promising and there's time to tune properly.
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
  
  # Known env gotcha: this xgboost version returns predict() already shaped
  # nrow x num_class — verify once on first fold rather than assuming.
  if (f == sort(unique(oof_all$fold))[1]) {
    cat("predict() class:", class(pred), " dim:", paste(dim(pred), collapse="x"), "\n")
  }
  if (!is.null(dim(pred))) {
    meta_oof[te, ] <- pred
  } else {
    meta_oof[te, ] <- matrix(pred, ncol = 4, byrow = TRUE)
  }
}

actual_mat <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
meta_cv <- mlogloss(actual_mat, meta_oof)
meta_cv  # compare to Model 23 (glmnet stack, 1.163684) and Model 29 (1.154955)