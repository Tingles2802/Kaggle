# 30-07_4_meta_ext_tune.R
# Requires (same session, in order): full chain through 29-07_20, then
# 30-07_1_knn_oof.R, then 30-07_3_meta_ext.R (gives d_train_ext, feat_cols_ext, fold_vec)
# fold_vec confirmed 2026-07-30 == train$fold (5 balanced folds, no Case leakage)
# Gotcha #29: xgb.cv() object has no $best_iteration (that's an xgb.train fit field) -
# derive best iter via which.min(evaluation_log$test_mlogloss_mean) instead.

folds_list <- lapply(1:5, function(k) which(fold_vec == k))

X_ext <- as.matrix(d_train_ext[, ..feat_cols_ext])
y_ext <- d_train_ext[, Ch2*1 + Ch3*2 + Ch4*3]  # 0-indexed label from Ch1..Ch4 actuals
dtrain_ext <- xgb.DMatrix(data = X_ext, label = y_ext)

grid <- expand.grid(
  max_depth = c(2, 3, 4),
  eta = c(0.02, 0.03, 0.05),
  subsample = 0.8,
  colsample_bytree = 0.8
)

results <- data.table()
for (i in 1:nrow(grid)) {
  params <- list(
    objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss",
    max_depth = grid$max_depth[i], eta = grid$eta[i],
    subsample = grid$subsample[i], colsample_bytree = grid$colsample_bytree[i]
  )
  cv <- xgb.cv(params = params, data = dtrain_ext, nrounds = 500,
               folds = folds_list, early_stopping_rounds = 20, verbose = 0)
  best_iter <- which.min(cv$evaluation_log$test_mlogloss_mean)
  best_score <- cv$evaluation_log$test_mlogloss_mean[best_iter]
  results <- rbind(results, data.table(
    max_depth = grid$max_depth[i], eta = grid$eta[i],
    nrounds = best_iter, cv_logloss = best_score))
  cat(sprintf("depth=%d eta=%.3f nrounds=%d CV=%.6f\n",
              grid$max_depth[i], grid$eta[i], best_iter, best_score))
}

results[order(cv_logloss)]