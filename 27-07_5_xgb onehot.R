# 27-07_5_xgb_onehot.R
# Track B follow-up: one-hot encode the 19 categorical attributes (x4 alts = 76 cols)
# for xgboost, which previously got them as raw integer codes (see experiment_log
# Section 2 note + Open Questions: "unclear how much this is holding it back").
# ranger already uses factors correctly (Step 3 of 27-07_4) — no change needed there.
#
# Run in order first: 27-07_1_pipeline_foundation.R, then 27-07_4_trees_setup.R
# (Steps 1-2 at least, for feature_cols/demo_cols/cat_feature_cols/X_cols/y_alt).
# This script does NOT re-run the integer-coded xgboost CV — that result (1.225570)
# is already logged.

# ---- Step 1: one-hot design matrix for categorical attributes only --------
# Price1-4 and demo_cols stay numeric (already sensible as-is); only the 76
# categorical attribute columns get expanded to dummies.
price_cols   <- paste0("Price", 1:4)
numeric_cols <- c(price_cols, demo_cols)

onehot_dt <- copy(train[, ..cat_feature_cols])
for (v in cat_feature_cols) onehot_dt[, (v) := as.factor(get(v))]

# Diagnostic: model.matrix(contrasts=FALSE) errors if any column has < 2 levels
# (contr.treatment requires >=2). Check + report which columns are degenerate —
# likely a rare attribute-alt combination that's constant (e.g. always "0"/inactive
# for that slot) — before building dummies.
n_levels <- sapply(onehot_dt, nlevels)
degenerate_cols <- names(n_levels)[n_levels < 2]
if (length(degenerate_cols) > 0) {
  cat("Degenerate (single-level) columns found, handled as constant dummy:\n")
  print(degenerate_cols)
}

# Manual dummy builder — full expansion per column (all levels, not levels-1;
# redundant for a linear model but harmless/standard for trees). Handles
# single-level columns without erroring (unlike model.matrix + contrasts=FALSE).
build_onehot <- function(dt, cols) {
  mats <- lapply(cols, function(v) {
    x  <- dt[[v]]
    lv <- levels(x)
    m  <- sapply(lv, function(l) as.integer(x == l))
    if (length(lv) == 1) m <- matrix(m, ncol = 1)  # sapply drops to vector when 1 level
    colnames(m) <- paste0(v, lv)
    m
  })
  do.call(cbind, mats)
}
onehot_mat <- build_onehot(onehot_dt, cat_feature_cols)

xgb_data_oh <- cbind(onehot_mat, as.matrix(train[, ..numeric_cols]))

cat("One-hot feature matrix:", ncol(xgb_data_oh), "columns (vs",
    length(X_cols), "integer-coded)\n")

# ---- Step 2: xgboost CV on one-hot features (same params as 27-07_4) ------
y_zero_indexed <- y_alt - 1L

cv_scores_xgb_oh <- numeric(5)
for (k in 1:5) {
  tr_idx <- which(train$fold != k)
  va_idx <- which(train$fold == k)
  
  dtrain <- xgb.DMatrix(data = xgb_data_oh[tr_idx, ], label = y_zero_indexed[tr_idx])
  dtest  <- xgb.DMatrix(data = xgb_data_oh[va_idx, ], label = y_zero_indexed[va_idx])
  
  fit_xgb_oh <- xgb.train(
    params = list(objective = "multi:softprob", num_class = 4,
                  eval_metric = "mlogloss", eta = 0.1, max_depth = 6),
    data = dtrain, nrounds = 200, verbose = 0
  )
  
  preds <- predict(fit_xgb_oh, dtest)   # nrow x 4 already (Gotcha #3 fix carried over — no reshape)
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  
  actual <- as.matrix(train[va_idx, .(Ch1, Ch2, Ch3, Ch4)])
  cv_scores_xgb_oh[k] <- mlogloss(actual, preds)
  cat("xgboost (one-hot) fold", k, "log-loss:", cv_scores_xgb_oh[k], "\n")
}

cat("\nxgboost (one-hot) CV — Mean:", mean(cv_scores_xgb_oh),
    " SD:", sd(cv_scores_xgb_oh), "\n")
cat("vs xgboost integer-coded: 1.225570\n")
cat("vs clogit baseline:       1.180861\n")