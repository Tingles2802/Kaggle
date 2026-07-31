# 31-07_1_lag_features_ab.R
# STANDALONE -- no prereqs, fresh fread.
#
# Purpose: cheap diagnostic to check if choice-inertia (lag of PREVIOUS TASK'S actual chosen
# alt, same Case) carries real signal, BEFORE investing in the recursive test-time pipeline
# (train has zero Case overlap with test -- confirmed -- so test-time lag isn't directly
# available; would need: predict Task 1 -> use soft prediction as pseudo-lag for Task 2 -> ...).
# This script uses TRUE lag (only valid for train/CV) to get an upper-bound signal estimate.
#
# If Delta here is small/noise -> drop the lever, don't build the recursive pipeline.
# If Delta is real (c.f. panel features' Delta=0.0034 threshold) -> proceed to recursive build.

library(data.table)
library(xgboost)

train <- fread("train.csv")
setorder(train, Case, Task)

# --- true lag features: previous task's chosen alt, within Case. Task==1 -> 0,0,0,0 ---
train[, `:=`(PrevCh1 = shift(Ch1, 1, fill = 0),
             PrevCh2 = shift(Ch2, 1, fill = 0),
             PrevCh3 = shift(Ch3, 1, fill = 0),
             PrevCh4 = shift(Ch4, 1, fill = 0)), by = Case]
stopifnot(all(train[Task == 1, PrevCh1 + PrevCh2 + PrevCh3 + PrevCh4] == 0))

# --- one-hot design per alt (Gotcha #20: drop single-level/constant attribute cols first) ---
attr_cols <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")

onehot_block <- function(alt) {
  cols <- c(paste0(attr_cols, alt), paste0("Price", alt))
  d <- copy(train[, ..cols])
  setnames(d, cols, c(attr_cols, "Price"))
  d[, (attr_cols) := lapply(.SD, as.factor), .SDcols = attr_cols]
  nlev <- sapply(d[, ..attr_cols], function(x) length(unique(x)))
  keep_cols <- names(nlev)[nlev > 1]   # drops Alt4's constant attrs
  if (length(keep_cols) == 0) {
    # Alt4: all attrs constant (structural opt-out) -- no dummy cols to build, Price only
    mm <- data.table(Price = d$Price)
  } else {
    mm <- as.data.table(model.matrix(~ . - 1, data = d[, ..keep_cols]))
    mm[, Price := d$Price]
  }
  setnames(mm, names(mm), paste0("A", alt, "_", names(mm)))
  mm
}

X_base <- cbind(onehot_block(1), onehot_block(2), onehot_block(3), onehot_block(4))
y <- train[, Ch2 * 1 + Ch3 * 2 + Ch4 * 3]  # 0-indexed label (Ch1=0,...,Ch4=3) for multi:softprob

X_noLag   <- as.matrix(X_base)
X_withLag <- as.matrix(cbind(X_base, train[, .(PrevCh1, PrevCh2, PrevCh3, PrevCh4)]))

# train$fold does NOT exist in the raw CSV -- it's normally added by 27-07_1 at session start.
# This is a fresh/standalone session (no 27-07_1 sourced), so derive it here: Case-grouped,
# 5-fold, fixed seed for internal reproducibility. NOTE: this will not exactly reproduce the
# canonical fold assignment from 27-07_1 (exact RNG sequence unknown), so the absolute No-lag
# CV number here may differ slightly from Model 20's logged 1.176378. That's fine for THIS
# diagnostic -- both No-lag and With-lag runs use the identical fold split below, so the Delta
# between them is the valid comparison, regardless of small absolute offset from canonical.
set.seed(2024)
case_ids <- unique(train$Case)
case_fold <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train <- merge(train, case_fold, by = "Case", sort = FALSE)
setorder(train, Case, Task)  # merge can reorder rows -- re-sort to match X_base's row order

folds_list <- lapply(1:5, function(k) which(train$fold == k))
stopifnot(all(lengths(folds_list) > 0))  # guard against the empty-fold bug that just occurred

params <- list(objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss",
               eta = 0.1, max_depth = 3)  # Model 20 params, for direct comparability

run_cv <- function(X, nrounds = 180) {
  dtrain <- xgb.DMatrix(data = X, label = y)
  cv <- xgb.cv(params = params, data = dtrain, nrounds = nrounds, folds = folds_list, verbose = 0)
  best_iter <- which.min(cv$evaluation_log$test_mlogloss_mean)  # Gotcha #29: no $best_iteration on xgb.cv()
  list(loss = cv$evaluation_log$test_mlogloss_mean[best_iter], iter = best_iter)
}

res_noLag   <- run_cv(X_noLag)
res_withLag <- run_cv(X_withLag)

cat("No-lag CV:       ", res_noLag$loss,   " (iter", res_noLag$iter, ")\n")
cat("True-lag CV:      ", res_withLag$loss, " (iter", res_withLag$iter, ")\n")
cat("Delta (no - lag): ", res_noLag$loss - res_withLag$loss, "\n")

# Paste back both CV numbers + Delta.