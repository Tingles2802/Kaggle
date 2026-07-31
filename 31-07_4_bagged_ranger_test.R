# REQUIRES: 27-07_1_pipeline_foundation.R then 27-07_4_trees_setup.R sourced first in this session
# (needs ranger_data, train w/ Ch1-4 + fold columns).
# Tests whether multi-seed bagging (averaging B ranger fits per fold) beats a single fit.
# Model 18 params: mtry=40, min.node.size=60, num.trees=500 (matches untuned spec's num.trees).

library(data.table)
library(ranger)

stopifnot(exists("ranger_data"), exists("train"))

mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmin(pmax(pred_mat, eps), 1 - eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

B <- 5              # number of bagged seeds per fold
ntree <- 500
mtry_val <- 40
min_node <- 60

folds <- sort(unique(ranger_data$fold))

single_seed_loss <- numeric(length(folds))
bagged_loss <- numeric(length(folds))

for (k in folds) {
  train_idx <- which(ranger_data$fold != k)
  test_idx  <- which(ranger_data$fold == k)

  train_fold <- ranger_data[train_idx][, !c("fold"), with = FALSE]  # keeps y + features
  test_fold  <- ranger_data[test_idx][, !c("fold"), with = FALSE]
  actual <- as.matrix(train[test_idx, .(Ch1, Ch2, Ch3, Ch4)])

  pred_stack <- vector("list", B)

  for (b in 1:B) {
    fit <- ranger(
      y ~ ., data = train_fold,
      probability = TRUE, mtry = mtry_val, min.node.size = min_node,
      num.trees = ntree, seed = 1000 + b
    )
    preds <- predict(fit, data = test_fold)$predictions  # cols ordered by factor levels "1".."4"
    colnames(preds) <- c("Ch1", "Ch2", "Ch3", "Ch4")
    pred_stack[[b]] <- preds
  }

  single_seed_loss[k] <- mlogloss(actual, pred_stack[[1]])

  bagged_pred <- Reduce("+", pred_stack) / B
  bagged_loss[k] <- mlogloss(actual, bagged_pred)
}

cat("Single-seed CV (this script's design):", mean(single_seed_loss), "sd:", sd(single_seed_loss), "\n")
cat("Bagged (B=", B, ") CV:", mean(bagged_loss), "sd:", sd(bagged_loss), "\n")
cat("Delta (single - bagged, positive = bagging helped):", mean(single_seed_loss) - mean(bagged_loss), "\n")
cat("Reference: logged Model 18 CV = 1.189566\n")
