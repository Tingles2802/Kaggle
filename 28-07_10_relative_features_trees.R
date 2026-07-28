# Relative-price features (PriceRank/PriceGap) added to ranger's feature set.
# Ranger only — xgboost part pending 27-07_18/19 (one-hot construction, not yet available).
# Requires: 27-07_1, then 27-07_4 sourced first (train, fold, attr_vars, mlogloss(), X_cols).

library(ranger)
library(data.table)

price_cols <- paste0("Price", 1:4)
price_mat  <- as.matrix(train[, ..price_cols])
price_rank <- t(apply(price_mat, 1, rank, ties.method = "min"))
price_min  <- apply(price_mat, 1, min)
colnames(price_rank) <- paste0("PriceRank", 1:4)
price_gap  <- price_mat - price_min
colnames(price_gap)  <- paste0("PriceGap", 1:4)

# Rebuild ranger_data exactly as in 27-07_4, then add rel features on top
cat_feature_cols <- unlist(lapply(1:4, function(a) paste0(setdiff(attr_vars, "Price"), a)))
ranger_data2 <- copy(train[, ..X_cols])
for (v in cat_feature_cols) ranger_data2[, (v) := as.factor(get(v))]
ranger_data2[, y := factor(train$chosen_alt, levels = 1:4)]
ranger_data2[, fold := train$fold]
ranger_data2 <- cbind(ranger_data2, as.data.table(price_rank), as.data.table(price_gap))

stopifnot(nrow(ranger_data2) == nrow(train))

# Model 18's tuned params (mtry=40, min.node.size=60), same 5-fold CV
cv_scores <- numeric(5)
for (k in 1:5) {
  tr <- ranger_data2[fold != k]
  va <- ranger_data2[fold == k]
  fit <- ranger(y ~ ., data = tr[, !c("fold"), with = FALSE],
                probability = TRUE, num.trees = 500, seed = 2024,
                mtry = 40, min.node.size = 60)
  preds <- predict(fit, data = va)$predictions
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  actual <- as.matrix(train[fold == k, .(Ch1, Ch2, Ch3, Ch4)])
  cv_scores[k] <- mlogloss(actual, preds)
  cat("fold", k, "log-loss:", cv_scores[k], "\n")
}
cat("\nranger+relprice CV — Mean:", mean(cv_scores), " SD:", sd(cv_scores),
    " (Model 18 baseline: 1.189566)\n")

library(xgboost)

# Rebuild one-hot categorical block exactly as in 27-07_5 (unchanged by this experiment)
onehot_dt <- copy(train[, ..cat_feature_cols])
for (v in cat_feature_cols) onehot_dt[, (v) := as.factor(get(v))]
build_onehot <- function(dt, cols) {
  mats <- lapply(cols, function(v) {
    x <- dt[[v]]; lv <- levels(x)
    m <- sapply(lv, function(l) as.integer(x == l))
    if (length(lv) == 1) m <- matrix(m, ncol = 1)
    colnames(m) <- paste0(v, lv)
    m
  })
  do.call(cbind, mats)
}
onehot_mat <- build_onehot(onehot_dt, cat_feature_cols)

price_cols   <- paste0("Price", 1:4)   # already have this from earlier, but confirm
numeric_cols <- c(price_cols, demo_cols)   # demo_cols must exist from 27-07_4 — check exists("demo_cols")

# numeric_cols widened: original (Price1-4 + demo_cols) + new PriceRank/PriceGap
numeric_mat2 <- cbind(as.matrix(train[, ..numeric_cols]), price_rank, price_gap)
xgb_data_oh2 <- cbind(onehot_mat, numeric_mat2)

y_zero_indexed <- y_alt - 1L
cat("xgb_data_oh2 columns:", ncol(xgb_data_oh2), "(vs", ncol(onehot_mat) + length(numeric_cols), "before)\n")

# Model 20's tuned params (eta=0.1, depth=3), same early-stopping CV as 27-07_18/19
cv_scores_xgb <- numeric(5)
for (k in 1:5) {
  tr_idx <- which(train$fold != k)
  va_idx <- which(train$fold == k)
  dtrain <- xgb.DMatrix(data = xgb_data_oh2[tr_idx, ], label = y_zero_indexed[tr_idx])
  dtest  <- xgb.DMatrix(data = xgb_data_oh2[va_idx, ], label = y_zero_indexed[va_idx])
  fit <- xgb.train(
    params = list(objective = "multi:softprob", num_class = 4,
                  eval_metric = "mlogloss", eta = 0.1, max_depth = 3),
    data = dtrain, nrounds = 1000, evals = list(val = dtest),
    early_stopping_rounds = 20, verbose = 0
  )
  best_iter <- as.integer(xgb.attr(fit, "best_iteration")) + 1L
  preds <- predict(fit, dtest, iterationrange = c(1L, best_iter))
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  actual <- as.matrix(train[va_idx, .(Ch1, Ch2, Ch3, Ch4)])
  cv_scores_xgb[k] <- mlogloss(actual, preds)
  cat("fold", k, "log-loss:", cv_scores_xgb[k], "\n")
}
cat("\nxgboost+relprice CV — Mean:", mean(cv_scores_xgb), " SD:", sd(cv_scores_xgb),
    " (Model 20 baseline: 1.176378)\n")