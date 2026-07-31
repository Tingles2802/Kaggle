# calibration_diagnostic.R
# Requires (in-memory, from Model 32 chain ending 29-07_20): oof_all, feat_cols, fold_vec, params_tuned, nrounds_tuned
# Gotcha #30: this xgboost version's xgb.cv() stores OOF preds at cv$cv_predict$pred (not cv$pred).
# No new randomness — folds are fixed via folds=, no set.seed needed.

library(xgboost)
library(data.table)

y_actual <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
y_idx    <- max.col(y_actual) - 1L
X        <- as.matrix(oof_all[, ..feat_cols])
dtrain_cal <- xgb.DMatrix(data = X, label = y_idx)

cv_cal <- xgb.cv(
  params = params_tuned,  # already has objective/num_class/eval_metric — don't re-add
  data = dtrain_cal,
  nrounds = nrounds_tuned,
  folds = lapply(1:5, function(k) which(fold_vec == k)),
  prediction = TRUE,
  verbose = 0
)

oof_pred <- cv_cal$cv_predict$pred
colnames(oof_pred) <- c("p1","p2","p3","p4")

# Sanity check: should match Model 32's logged CV (~1.151683 / fresh-seed 1.152654)
cat("Sanity logloss:", mlogloss(y_actual, oof_pred), "\n")

# A) Aggregate bias per class
agg_check <- data.table(class = 1:4, mean_pred = colMeans(oof_pred), actual_share = colMeans(y_actual))
agg_check[, diff := mean_pred - actual_share]
print(agg_check)

# B) Per-class reliability (decile bins)
reliability <- function(p, actual, nbins = 10) {
  bins <- cut(p, breaks = quantile(p, probs = seq(0,1,length.out = nbins+1)), include.lowest = TRUE, labels = FALSE)
  data.table(bin = bins, p = p, actual = actual)[, .(mean_pred = mean(p), mean_actual = mean(actual), n = .N), by = bin][order(bin)]
}
for (c in 1:4) { cat("--- Class", c, "---\n"); print(reliability(oof_pred[,c], y_actual[,c])) }

# C) Top-1 confidence vs accuracy
conf <- apply(oof_pred, 1, max)
pred_class <- max.col(oof_pred)
correct <- as.integer(pred_class == (y_idx + 1L))
conf_bins <- cut(conf, breaks = quantile(conf, probs = seq(0,1,length.out = 11)), include.lowest = TRUE, labels = FALSE)
conf_reliability <- data.table(bin = conf_bins, conf = conf, correct = correct)[, .(mean_conf = mean(conf), accuracy = mean(correct), n = .N), by = bin][order(bin)]
print(conf_reliability)

# D) Expected Calibration Error (top-1)
ece <- sum(conf_reliability$n / sum(conf_reliability$n) * abs(conf_reliability$mean_conf - conf_reliability$accuracy))
cat("ECE (top-1):", ece, "\n")