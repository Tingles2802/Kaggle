# 27-07_16_ranger_tuning.R
# REQUIRES: rm(list=ls()), then source 27-07_1, then 27-07_4 (need ranger_data, X_cols, mlogloss(), fold).
# Verify first: re-run 27-07_4's ranger CV block, confirm 1.218629 before trusting anything below.

library(ranger)

# grid: mtry (default sqrt(p) ~ 9 for ~80+11 cols) and min.node.size
p <- length(X_cols)
grid <- expand.grid(
  mtry = c(round(sqrt(p)), round(p/6), round(p/4)),
  min.node.size = c(1, 5, 10, 20)
)

grid$cv_mean <- NA_real_
grid$cv_sd   <- NA_real_

for (g in seq_len(nrow(grid))) {
  fold_scores <- numeric(5)
  for (k in 1:5) {
    train_k <- ranger_data[fold != k]
    test_k  <- ranger_data[fold == k]

    fit_rf <- ranger(
      y ~ ., data = train_k[, !c("fold"), with = FALSE],
      probability = TRUE, num.trees = 500, seed = 2024,
      mtry = grid$mtry[g], min.node.size = grid$min.node.size[g]
    )
    preds <- predict(fit_rf, data = test_k)$predictions
    colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")

    actual <- as.matrix(train[fold == k, .(Ch1, Ch2, Ch3, Ch4)])
    fold_scores[k] <- mlogloss(actual, preds)
  }
  grid$cv_mean[g] <- mean(fold_scores)
  grid$cv_sd[g]   <- sd(fold_scores)
  cat("mtry:", grid$mtry[g], "min.node.size:", grid$min.node.size[g],
      "-> mean:", grid$cv_mean[g], "sd:", grid$cv_sd[g], "\n")
}

grid <- grid[order(grid$cv_mean), ]
print(grid)
cat("\nBest vs Model 7 baseline (1.218629):", grid$cv_mean[1], "\n")
