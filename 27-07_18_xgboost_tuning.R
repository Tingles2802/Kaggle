# 27-07_18_xgboost_tuning.R
# REQUIRES (run first, in order): 27-07_1_pipeline_foundation.R,
#   27-07_4_trees_setup.R (Steps 1-2 only, for feature_cols/demo_cols/
#   cat_feature_cols/X_cols/y_alt), 27-07_5_xgb_onehot.R (for xgb_data_oh,
#   numeric_cols, y_zero_indexed).
# Does NOT require 27-07_2/3/6-17 (clogit/ranger scripts) — verify with
# ls() if unsure; don't assume, check (Gotcha #14/15).
#
# xgboost 3.2.1.1 gotcha (found this session): Booster is now an external
# pointer object — names(fit) is just "ptr". $best_iteration/$evaluation_log
# no longer exist as list fields. Use xgb.attr(fit, "best_iteration")
# (0-indexed) instead. evals= (not watchlist=) is required.

# ---- Grid: eta x max_depth, nrounds chosen per fold via early stopping ----
grid <- expand.grid(eta = c(0.05, 0.1, 0.2), max_depth = c(4, 6, 8))
grid$cv_mean <- NA_real_
grid$cv_sd   <- NA_real_
grid$mean_best_iter <- NA_real_

for (g in 1:nrow(grid)) {
  eta_g   <- grid$eta[g]
  depth_g <- grid$max_depth[g]
  
  fold_scores <- numeric(5)
  fold_iters  <- numeric(5)
  
  for (k in 1:5) {
    tr_idx <- which(train$fold != k)
    va_idx <- which(train$fold == k)
    
    dtrain <- xgb.DMatrix(data = xgb_data_oh[tr_idx, ], label = y_zero_indexed[tr_idx])
    dtest  <- xgb.DMatrix(data = xgb_data_oh[va_idx, ], label = y_zero_indexed[va_idx])
    
    fit <- xgb.train(
      params = list(objective = "multi:softprob", num_class = 4,
                    eval_metric = "mlogloss", eta = eta_g, max_depth = depth_g),
      data = dtrain, nrounds = 1000,
      evals = list(val = dtest),
      early_stopping_rounds = 20, verbose = 0
    )
    
    best_iter_0idx <- as.integer(xgb.attr(fit, "best_iteration"))  # 0-indexed (xgboost 3.x)
    best_iter <- best_iter_0idx + 1L                                # 1-indexed round count, for reporting
    preds <- predict(fit, dtest, iterationrange = c(1L, best_iter))
    colnames(preds) <- c("Ch1", "Ch2", "Ch3", "Ch4")
    
    actual <- as.matrix(train[va_idx, .(Ch1, Ch2, Ch3, Ch4)])
    fold_scores[k] <- mlogloss(actual, preds)
    fold_iters[k]  <- best_iter
  }
  
  grid$cv_mean[g]       <- mean(fold_scores)
  grid$cv_sd[g]         <- sd(fold_scores)
  grid$mean_best_iter[g] <- mean(fold_iters)
  
  cat("eta=", eta_g, " depth=", depth_g,
      " CV mean=", round(grid$cv_mean[g], 6),
      " sd=", round(grid$cv_sd[g], 6),
      " mean_best_iter=", round(grid$mean_best_iter[g], 1), "\n", sep = "")
}

grid <- grid[order(grid$cv_mean), ]
print(grid)
cat("\nBest combo:\n")
print(grid[1, ])
cat("\nvs xgboost one-hot untuned: 1.220720\n")
cat("vs clogit best (Model 15): 1.162469\n")