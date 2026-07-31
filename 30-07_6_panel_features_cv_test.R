# Prerequisites: 27-07_1, 27-07_4 (creates `train`, `ranger_data` w/ y+fold cols),
# and 30-07_5_panel_features.R already run (needs case_panel_features_train.csv on disk).
# Purpose: fast A/B signal check only - NOT meant to reproduce logged Model 20 CV exactly
# (uses a fresh one-hot design, not the canonical xgb_data_oh from 28-07_1, which isn't
# available here). Just checks: do panel features move CV at all, holding design fixed.

library(data.table)
library(xgboost)

panel <- fread("case_panel_features_train.csv")
stopifnot(nrow(panel) == nrow(train), all(panel$No == train$No))  # alignment guard

attr_cols <- setdiff(names(ranger_data), c("y","fold"))
is_constant <- sapply(ranger_data[, ..attr_cols], function(x) length(unique(x)) < 2)  # Gotcha #20
attr_cols <- attr_cols[!is_constant]
cat("Dropped", sum(is_constant), "constant cols:", names(is_constant)[is_constant], "\n")

X_base <- model.matrix(~ . - 1, data = ranger_data[, ..attr_cols])
X_ext  <- cbind(X_base, as.matrix(panel[, .(PriceDev1, PriceDev2, PriceDev3,
                                            case_mean_price, case_sd_price,
                                            TaskFrac, IsFirstTask)]))

y0 <- as.integer(ranger_data$y) - 1L
folds_list <- lapply(1:5, function(k) which(ranger_data$fold == k))

params <- list(objective = "multi:softprob", num_class = 4,
               eval_metric = "mlogloss", eta = 0.1, max_depth = 3)

run_cv <- function(X) {
  dtrain <- xgb.DMatrix(data = X, label = y0)
  cv <- xgb.cv(params = params, data = dtrain, nrounds = 200,
               folds = folds_list, early_stopping_rounds = 20, verbose = 0)
  best_i <- which.min(cv$evaluation_log$test_mlogloss_mean)  # Gotcha #29
  cv$evaluation_log$test_mlogloss_mean[best_i]
}

cv_base <- run_cv(X_base)
cv_ext  <- run_cv(X_ext)

cat("CV without panel features:", cv_base, "\n")
cat("CV with panel features:   ", cv_ext, "\n")
cat("Delta:", cv_base - cv_ext, "(positive = improvement)\n")