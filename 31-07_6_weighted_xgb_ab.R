# STANDALONE - no other R files required. Fresh fread, own fold split (Gotcha #32).
# Tests segment/inverse-propensity row weighting on Model 20's xgboost spec.
# Must reproduce baseline CV 1.176378 first (diagnostic-first rule) before trusting weighted result.

library(data.table)
library(xgboost)

train <- fread("train.csv")

# --- Case-grouped fold split (seed 2024, matches original train$fold) ---
set.seed(2024)
cases <- unique(train$Case)
case_fold <- data.table(Case = cases, fold = sample(1:5, length(cases), replace = TRUE))
train <- merge(train, case_fold, by = "Case")

# --- segment -> test/train share weight ---
seg_weight <- data.table(
  segment = c("Small Car", "Midsize Car", "Midsize Luxury Utility segements",
              "Midsize Utility", "Prestige Luxury Sedan", "Full-size Pickup"),
  w = c(0.000, 0.101, 7.419, 0.328, 7.235, 1.920)
)
train <- merge(train, seg_weight, by = "segment", all.x = TRUE)
stopifnot(!any(is.na(train$w)))  # every row must get a weight - stop if segment strings didn't match

# --- one-hot design (19 attrs x4 alts + Price x4), Model 20 spec ---
attr_cols <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")
build_X <- function(alt) {
  cols <- paste0(attr_cols, alt)
  d <- train[, ..cols]
  d[] <- lapply(d, as.factor)
  keep_cols <- names(d)[sapply(d, function(x) nlevels(x) > 1)]  # drop constant cols (Gotcha #20)
  if (length(keep_cols) == 0) {
    mm <- matrix(nrow = nrow(d), ncol = 0)  # Alt4: all attrs constant (Gotcha #33)
  } else {
    mm <- model.matrix(~ . - 1, data = d[, ..keep_cols])
  }
  cbind(mm, Price = train[[paste0("Price", alt)]])
}
X <- do.call(cbind, lapply(1:4, build_X))
y <- max.col(train[, .(Ch1, Ch2, Ch3, Ch4)]) - 1  # 0-indexed for xgboost

mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}
actual_mat <- as.matrix(train[, .(Ch1, Ch2, Ch3, Ch4)])

params <- list(objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss",
               eta = 0.1, max_depth = 3)
nrounds <- 180

run_cv <- function(weights = NULL) {
  oof <- matrix(0, nrow(train), 4)
  for (k in 1:5) {
    tr_idx <- which(train$fold != k); te_idx <- which(train$fold == k)
    dtrain <- xgb.DMatrix(X[tr_idx, ], label = y[tr_idx],
                          weight = if (is.null(weights)) NULL else weights[tr_idx])
    dtest <- xgb.DMatrix(X[te_idx, ])
    fit <- xgb.train(params, dtrain, nrounds = nrounds, verbose = 0)
    oof[te_idx, ] <- predict(fit, dtest)
  }
  mlogloss(actual_mat, oof)
}

cv_baseline <- run_cv(weights = NULL)
cat("Baseline CV (expect ~1.176378):", cv_baseline, "\n")

cv_weighted <- run_cv(weights = train$w)
cat("Weighted CV:", cv_weighted, "\n")
cat("Delta (weighted - baseline):", cv_weighted - cv_baseline, "\n")

# ALSO compute test-share-reweighted score of the WEIGHTED model's OOF (route-b style check):
# i.e. does the weighted model's own reweighted-eval CV move closer to actual LB (~1.196)?
seg_eval_w <- merge(data.table(idx = 1:nrow(train), segment = train$segment), seg_weight, by = "segment")
setorder(seg_eval_w, idx)
# (paste back OOF matrices from run_cv here if per-segment breakdown is wanted - not built into
# run_cv() above to keep this script minimal; ask Claude to extend if needed)