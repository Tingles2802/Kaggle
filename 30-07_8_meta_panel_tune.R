# Prerequisites: same session as 30-07_7_meta_panel_test.R (needs dtrain_ext2,
# folds_list in memory).
# Purpose: depth/eta grid search on the 35-col (28+panel) feature set, since
# best nrounds shifted vs Model 37 - Model 37's params may not be optimal here.

library(xgboost)

grid <- expand.grid(max_depth = c(2,3,4), eta = c(0.01,0.02,0.03,0.05))
results <- data.table::data.table()

for (i in 1:nrow(grid)) {
  params_i <- list(objective = "multi:softprob", num_class = 4,
                     eval_metric = "mlogloss",
                     max_depth = grid$max_depth[i], eta = grid$eta[i])
  cv_i <- xgb.cv(params = params_i, data = dtrain_ext2, nrounds = 600,
                 folds = folds_list, early_stopping_rounds = 30, verbose = 0)
  best_i <- which.min(cv_i$evaluation_log$test_mlogloss_mean)
  results <- rbind(results, data.table::data.table(
    max_depth = grid$max_depth[i], eta = grid$eta[i],
    best_nrounds = best_i,
    cv_loss = cv_i$evaluation_log$test_mlogloss_mean[best_i]
  ))
}

results <- results[order(cv_loss)]
print(results)
cat("\nBest config:\n")
print(results[1])
