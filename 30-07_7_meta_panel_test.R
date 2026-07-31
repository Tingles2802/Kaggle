# Prerequisites: 30-07_3_meta_ext.R, 30-07_4_meta_ext_tune.R (needs d_train_ext,
# feat_cols_ext, fold_vec, folds_list in memory), and 30-07_5_panel_features.R
# already run (needs case_panel_features_train.csv on disk).
# Purpose: quick check - does appending panel features to Model 37's meta-learner
# feature set help, using Model 37's already-tuned hyperparams (no re-grid yet).

library(data.table)
library(xgboost)

panel <- fread("case_panel_features_train.csv")

d_train_ext2 <- merge(d_train_ext, panel[, .(Case, Task, PriceDev1, PriceDev2, PriceDev3,
                                              case_mean_price, case_sd_price,
                                              TaskFrac, IsFirstTask)],
                       by = c("Case","Task"), all.x = TRUE)
stopifnot(nrow(d_train_ext2) == nrow(d_train_ext))  # merge didn't drop/dup rows

feat_cols_ext2 <- c(feat_cols_ext, "PriceDev1","PriceDev2","PriceDev3",
                     "case_mean_price","case_sd_price","TaskFrac","IsFirstTask")

X_ext2 <- as.matrix(d_train_ext2[, ..feat_cols_ext2])
dtrain_ext2 <- xgb.DMatrix(data = X_ext2, label = y_ext)

params_37 <- list(objective = "multi:softprob", num_class = 4,
                    eval_metric = "mlogloss", eta = 0.02, max_depth = 3)

cv38 <- xgb.cv(params = params_37, data = dtrain_ext2, nrounds = 400,
               folds = folds_list, early_stopping_rounds = 30, verbose = 0)
best_i <- which.min(cv38$evaluation_log$test_mlogloss_mean)  # Gotcha #29

cat("Model 37 CV (28-col, logged):        1.150705\n")
cat("Model 38 candidate CV (35-col):      ", cv38$evaluation_log$test_mlogloss_mean[best_i], "\n")
cat("Best nrounds:", best_i, "\n")
