# Uses in-memory oof_all, fold_vec, feat_cols from the sourced chain. Needs oof_nnet, oof_knn (in-memory or CSV).
library(data.table); library(xgboost)

if (!exists("oof_nnet")) oof_nnet <- fread("oof_nnet.csv")
if (!exists("oof_knn"))  oof_knn  <- fread("oof_knn.csv")

oof_knn <- fread("oof_knn.csv")
d_train_ext <- merge(oof_all, oof_nnet, by = c("Case","Task"))
d_train_ext <- merge(d_train_ext, oof_knn, by = c("Case","Task"))
stopifnot(nrow(d_train_ext) == 21565)  # guard against join row loss

feat_cols_ext <- c(feat_cols, paste0("nnet_Ch", 1:4), paste0("knn_Ch", 1:4))

mlogloss <- function(actual, pred, eps = 1e-15) {
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(actual * log(pred)))
}

params_tuned <- list(max_depth = 3, eta = 0.03, subsample = 0.8, colsample_bytree = 0.8,
                     objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss")
nrounds_tuned <- 200

y_lab <- max.col(as.matrix(d_train_ext[, .(Ch1, Ch2, Ch3, Ch4)])) - 1  # 0-indexed for xgboost
oof_meta_ext <- matrix(NA_real_, nrow(d_train_ext), 4)

for (i in sort(unique(fold_vec))) {
  tr <- fold_vec != i; te <- fold_vec == i
  dtr <- xgb.DMatrix(as.matrix(d_train_ext[tr, ..feat_cols_ext]), label = y_lab[tr])
  dte <- xgb.DMatrix(as.matrix(d_train_ext[te, ..feat_cols_ext]))
  fit <- xgb.train(params_tuned, dtr, nrounds = nrounds_tuned)
  oof_meta_ext[te, ] <- predict(fit, dte)
}

actual_mat <- as.matrix(d_train_ext[, .(Ch1, Ch2, Ch3, Ch4)])
cat("Extended meta-learner CV:", mlogloss(actual_mat, oof_meta_ext), "vs Model 32 baseline 1.151683\n")