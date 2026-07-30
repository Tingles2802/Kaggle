# Requires: 29-07_2 then 29-07_10 sourced first. Needs oof_nnet.csv, oof_knn.csv in wd.
library(data.table); library(xgboost)

oof_nnet <- fread("oof_nnet.csv")
oof_knn  <- fread("oof_knn.csv")

# TODO: confirm d_meta_train's actual column structure before this merge - not verified
# in the log. If this errors, paste str(d_meta_train)/colnames(d_meta_train) back.
d_meta_train_ext <- merge(d_meta_train, oof_nnet, by = c("Case","Task"))
d_meta_train_ext <- merge(d_meta_train_ext, oof_knn, by = c("Case","Task"))

feat_cols_ext <- c(feat_cols, paste0("nnet_Ch", 1:4), paste0("knn_Ch", 1:4))

params_tuned <- list(max_depth = 3, eta = 0.03, subsample = 0.8, colsample_bytree = 0.8,
                     objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss")
nrounds_tuned <- 200

folds <- train$fold
oof_meta_ext <- matrix(NA_real_, nrow(train), 4)

for (i in sort(unique(folds))) {
  tr <- folds != i; te <- folds == i
  dtr <- xgb.DMatrix(as.matrix(d_meta_train_ext[tr, ..feat_cols_ext]), label = y_zero_indexed[tr])
  dte <- xgb.DMatrix(as.matrix(d_meta_train_ext[te, ..feat_cols_ext]))
  fit <- xgb.train(params_tuned, dtr, nrounds = nrounds_tuned)
  oof_meta_ext[te, ] <- predict(fit, dte)
}

actual_mat <- as.matrix(train[, .(Ch1, Ch2, Ch3, Ch4)])
cat("Extended meta-learner CV:", mlogloss(actual_mat, oof_meta_ext), "vs Model 32 baseline 1.151683\n")