# Requires: 27-07_1, 27-07_4 sourced first. Do not run via run_all_prereqs.R this session.
library(data.table); library(class)

feat_dt <- copy(ranger_data)
feat_dt[, c("y", "fold") := NULL]

# Gotcha #20: drop constant columns (Alt 4 structural zeros) before model.matrix
is_constant <- sapply(feat_dt, function(col) length(unique(col)) <= 1)
feat_dt <- feat_dt[, .SD, .SDcols = !is_constant]

X <- model.matrix(~ . , data = feat_dt)[, -1]  # dummy-encode factors, drop intercept
X <- scale(X)

y <- as.integer(ranger_data$y)          # 1-4 class labels
folds <- ranger_data$fold
actual_mat <- as.matrix(train[, .(Ch1, Ch2, Ch3, Ch4)])  # row-aligned to ranger_data

mlogloss <- function(actual, pred, eps = 1e-15) {
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(actual * log(pred)))
}

knn_prob <- function(tr_x, tr_y, te_x, k) {
  pred <- knn(tr_x, te_x, cl = as.factor(tr_y), k = k, prob = TRUE)
  won <- attr(pred, "prob")
  cls <- as.integer(as.character(pred))
  m <- matrix((1 - won) / 3, length(pred), 4)
  for (i in seq_along(pred)) m[i, cls[i]] <- won[i]
  m
}

k_grid <- c(15, 25, 35, 50)
results <- data.table(k = k_grid, cv_logloss = NA_real_)
oof_by_k <- list()

for (kk in k_grid) {
  oof <- matrix(NA_real_, nrow(X), 4)
  for (i in sort(unique(folds))) {
    tr <- folds != i; te <- folds == i
    oof[te, ] <- knn_prob(X[tr, ], y[tr], X[te, ], kk)
  }
  oof_by_k[[as.character(kk)]] <- oof
  results[k == kk, cv_logloss := mlogloss(actual_mat, oof)]
}
print(results)

best_k <- results[which.min(cv_logloss), k]
oof_knn <- as.data.table(oof_by_k[[as.character(best_k)]])
setnames(oof_knn, paste0("knn_Ch", 1:4))
fwrite(cbind(train[, .(Case, Task)], oof_knn), "oof_knn.csv")