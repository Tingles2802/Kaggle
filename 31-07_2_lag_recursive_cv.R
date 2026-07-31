# 31-07_2_lag_recursive_cv.R
# STANDALONE -- no prereqs, fresh fread. rm(list=ls()) first if reusing a session.
#
# Purpose: honest CV for choice-inertia. 31-07_1's Delta=0.0760 used TRUE previous-task labels,
# which don't exist at test time (test Cases are fully unseen -- zero Case overlap w/ train).
# This script simulates the REAL deployment: for each fold, train on true lag (legitimate --
# those Cases' full task sequences are in-fold), then on the held-out Cases predict task-by-task
# IN ORDER, feeding each task's own soft prediction forward as next task's lag feature. This
# mirrors exactly what the test-set submission pipeline will have to do.

library(data.table)
library(xgboost)

train <- fread("train.csv")
setorder(train, Case, Task)

train[, `:=`(PrevCh1 = shift(Ch1, 1, fill = 0),
             PrevCh2 = shift(Ch2, 1, fill = 0),
             PrevCh3 = shift(Ch3, 1, fill = 0),
             PrevCh4 = shift(Ch4, 1, fill = 0)), by = Case]
stopifnot(all(train[Task == 1, PrevCh1 + PrevCh2 + PrevCh3 + PrevCh4] == 0))

attr_cols <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")

onehot_block <- function(alt) {
  cols <- c(paste0(attr_cols, alt), paste0("Price", alt))
  d <- copy(train[, ..cols])
  setnames(d, cols, c(attr_cols, "Price"))
  d[, (attr_cols) := lapply(.SD, as.factor), .SDcols = attr_cols]
  nlev <- sapply(d[, ..attr_cols], function(x) length(unique(x)))
  keep_cols <- names(nlev)[nlev > 1]
  if (length(keep_cols) == 0) {
    mm <- data.table(Price = d$Price)          # Alt4: all attrs constant
  } else {
    mm <- as.data.table(model.matrix(~ . - 1, data = d[, ..keep_cols]))
    mm[, Price := d$Price]
  }
  setnames(mm, names(mm), paste0("A", alt, "_", names(mm)))
  mm
}

X_base <- as.matrix(cbind(onehot_block(1), onehot_block(2), onehot_block(3), onehot_block(4)))
y <- train[, Ch2 * 1 + Ch3 * 2 + Ch4 * 3]          # 0-indexed label
Y_actual <- as.matrix(train[, .(Ch1, Ch2, Ch3, Ch4)])

set.seed(2024)
case_ids <- unique(train$Case)
case_fold <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train <- merge(train, case_fold, by = "Case", sort = FALSE)
setorder(train, Case, Task)                        # re-sync order with X_base/y
stopifnot(all(diff(which(!duplicated(train$Case))) > 0))  # sanity: still Case-Task sorted

params <- list(objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss",
               eta = 0.1, max_depth = 3)
NROUNDS <- 176   # best_iter from 31-07_1's true-lag run; starting point, not re-tuned here

mlogloss <- function(actual, pred, eps = 1e-15) {
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(actual * log(pred)))
}

max_task <- max(train$Task)
oof_pred <- matrix(NA_real_, nrow = nrow(train), ncol = 4)

for (k in 1:5) {
  train_idx <- which(train$fold != k)
  test_idx  <- which(train$fold == k)

  dtr <- xgb.DMatrix(cbind(X_base[train_idx, ], as.matrix(train[train_idx, .(PrevCh1, PrevCh2, PrevCh3, PrevCh4)])),
                      label = y[train_idx])
  fit <- xgb.train(params, dtr, nrounds = NROUNDS)

  test_data <- train[test_idx]
  # running soft-lag table per held-out Case, updated task by task
  prev_dt <- data.table(Case = unique(test_data$Case), PrevCh1 = 0, PrevCh2 = 0, PrevCh3 = 0, PrevCh4 = 0)
  setkey(prev_dt, Case)

  for (t in 1:max_task) {
    is_t <- test_data$Task == t
    if (!any(is_t)) next
    rows_t <- test_idx[is_t]              # global row indices, aligned to X_base
    case_t <- test_data$Case[is_t]
    lag_t  <- as.matrix(prev_dt[J(case_t), .(PrevCh1, PrevCh2, PrevCh3, PrevCh4)])

    Xt <- xgb.DMatrix(cbind(X_base[rows_t, ], lag_t))
    p  <- predict(fit, Xt)                # already nrow x 4 on this xgboost version (no reshape= needed)
    oof_pred[rows_t, ] <- p

    upd <- data.table(Case = case_t, PrevCh1 = p[, 1], PrevCh2 = p[, 2], PrevCh3 = p[, 3], PrevCh4 = p[, 4])
    prev_dt[upd, on = "Case", `:=`(PrevCh1 = i.PrevCh1, PrevCh2 = i.PrevCh2, PrevCh3 = i.PrevCh3, PrevCh4 = i.PrevCh4)]
  }
  cat("fold", k, "done\n")
}

stopifnot(!anyNA(oof_pred))
recursive_cv <- mlogloss(Y_actual, oof_pred)
cat("Recursive (honest) lag CV:", recursive_cv, "\n")
cat("Compare to: No-lag CV 1.178917 (31-07_1) and leaky True-lag CV 1.102892 (31-07_1)\n")

# Paste back: the "Recursive (honest) lag CV" number.
