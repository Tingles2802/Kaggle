# 27-07_4_trees_setup.R
# Track B: ranger + xgboost, built directly on wide `train` (already one row
# per choice task with alt-specific columns) — no long/wide reshape needed.
# Run 27-07_1_pipeline_foundation.R first (needs train, fold, mlogloss()).

# ---- Step 1: confirm packages (STOP here if any FALSE) ----
pkg_check <- sapply(c("ranger", "xgboost", "glmnet"), requireNamespace, quietly = TRUE)
print(pkg_check)
if (!all(pkg_check[c("ranger", "xgboost")])) {
  stop("ranger/xgboost not available — install before proceeding, don't guess around this.")
}

library(ranger)
library(xgboost)

# ---- Step 2: feature set ----
# attr_vars includes Price already (see pipeline_foundation.R); 20 cols x 4 alts = 80
feature_cols <- unlist(lapply(1:4, function(a) paste0(attr_vars, a)))
demo_cols    <- c("segmentind", "yearind", "pparkind", "genderind",
                  "educind", "regionind", "Urbind",
                  "agea", "milesa", "nighta", "incomea")
stopifnot(all(c(feature_cols, demo_cols) %in% names(train)))

X_cols <- c(feature_cols, demo_cols)
y_alt  <- train$chosen_alt   # 1-4, built in pipeline_foundation.R

# ---- Step 3: ranger CV (categorical attrs as factors) ----
cat_feature_cols <- unlist(lapply(1:4, function(a) paste0(setdiff(attr_vars, "Price"), a)))

ranger_data <- copy(train[, ..X_cols])
for (v in cat_feature_cols) ranger_data[, (v) := as.factor(get(v))]
ranger_data[, y := factor(y_alt, levels = 1:4)]
ranger_data[, fold := train$fold]

cv_scores_ranger <- numeric(5)
for (k in 1:5) {
  train_k <- ranger_data[fold != k]
  test_k  <- ranger_data[fold == k]
  
  fit_rf <- ranger(
    y ~ ., data = train_k[, !c("fold"), with = FALSE],
    probability = TRUE, num.trees = 500, seed = 2024
  )
  preds <- predict(fit_rf, data = test_k)$predictions  # columns already ordered "1","2","3","4"
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  
  actual <- as.matrix(train[fold == k, .(Ch1, Ch2, Ch3, Ch4)])
  cv_scores_ranger[k] <- mlogloss(actual, preds)
  cat("ranger fold", k, "log-loss:", cv_scores_ranger[k], "\n")
}
cat("\nranger CV — Mean:", mean(cv_scores_ranger), " SD:", sd(cv_scores_ranger), "\n")

# ---- Step 4: xgboost CV (integer-coded categoricals fed directly — approximation,
#      not one-hot; acceptable for a first tree pass, revisit if this underperforms) ----
xgb_data <- as.matrix(train[, ..X_cols])
y_zero_indexed <- y_alt - 1L

cv_scores_xgb <- numeric(5)
for (k in 1:5) {
  tr_idx <- which(train$fold != k)
  va_idx <- which(train$fold == k)
  
  dtrain <- xgb.DMatrix(data = xgb_data[tr_idx, ], label = y_zero_indexed[tr_idx])
  dtest  <- xgb.DMatrix(data = xgb_data[va_idx, ], label = y_zero_indexed[va_idx])
  
  fit_xgb <- xgb.train(
    params = list(objective = "multi:softprob", num_class = 4,
                  eval_metric = "mlogloss", eta = 0.1, max_depth = 6),
    data = dtrain, nrounds = 200, verbose = 0
  )
  
  preds <- predict(fit_xgb, dtest)   # already nrow x 4, no reshape needed
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  
  actual <- as.matrix(train[va_idx, .(Ch1, Ch2, Ch3, Ch4)])
  cv_scores_xgb[k] <- mlogloss(actual, preds)
  cat("xgboost fold", k, "log-loss:", cv_scores_xgb[k], "\n")
}
cat("\nxgboost CV — Mean:", mean(cv_scores_xgb), " SD:", sd(cv_scores_xgb), "\n")
cat("\nxgboost CV complete.\n")