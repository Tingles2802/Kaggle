# Prerequisites: same session as 30-07_7/8 (needs dtrain_ext2, folds_list in memory).
# Purpose: fresh-seed stability check, same pattern as Model 32/37 (different xgb
# random seed, same folds/features/params) - confirms CV isn't a lucky fit.

library(xgboost)

params_38 <- list(objective = "multi:softprob", num_class = 4,
                    eval_metric = "mlogloss", eta = 0.02, max_depth = 3)

set.seed(99)
cv_fresh <- xgb.cv(params = params_38, data = dtrain_ext2, nrounds = 286,
                    folds = folds_list, verbose = 0)

cat("Model 38 CV (seed 2024, logged):  1.148969\n")
cat("Model 38 CV (fresh seed 99):     ",
    cv_fresh$evaluation_log$test_mlogloss_mean[286], "\n")
