# 29-07_19_nnet_full.R — Section 4a item 2: nnet::multinom, full attribute+Price+demo feature set
# STANDALONE — no prerequisite scripts needed. Reads train.csv directly, defines own mlogloss().
# Purpose: base-model diversity feature for next meta-learner iteration (not expected to beat clogit/xgb solo).

library(data.table)
library(nnet)

train <- fread("train.csv")

# mlogloss: wide n x 4 matrices, Ch1..Ch4 order, eps clipping
mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmin(pmax(pred_mat, eps), 1 - eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

# --- Reconstruct standard 5-fold, Case-grouped, seed-2024 split ---
set.seed(2024)
cases <- unique(train$Case)
case_fold <- data.table(Case = cases, fold = sample(rep(1:5, length.out = length(cases))))
train <- merge(train, case_fold, by = "Case", all.x = TRUE)

# --- Feature set: 19 attrs x4 (numeric) + Price x4 (numeric) + demo cols (factor) ---
attr_names <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")
attr_cols  <- as.vector(outer(attr_names, 1:4, paste0))
price_cols <- paste0("Price", 1:4)
demo_cols  <- c("segment","year","miles","night","ppark","gender","age","educ","region","Urb","income")

feat_cols <- c(attr_cols, price_cols, demo_cols)
for (col in demo_cols) train[[col]] <- factor(train[[col]])

# --- Outcome as single factor (1-4) from Ch1..Ch4 wide cols ---
train[, y := factor(max.col(as.matrix(.SD)), levels = 1:4), .SDcols = c("Ch1","Ch2","Ch3","Ch4")]

f <- as.formula(paste("y ~", paste(feat_cols, collapse = " + ")))

# --- 5-fold CV ---
fold_losses <- numeric(5)
oof_preds <- matrix(NA_real_, nrow = nrow(train), ncol = 4)

for (k in 1:5) {
  tr <- train[fold != k]
  te <- train[fold == k]
  
  fit <- multinom(f, data = tr, maxit = 300, trace = FALSE)
  pred <- predict(fit, newdata = te, type = "probs")
  
  # multinom drops a class column if only 3 levels present in tr$y — guard
  if (is.null(dim(pred))) stop("predict() returned a vector, not a matrix — check factor levels in fold ", k)
  if (ncol(pred) != 4) stop("predict() returned ", ncol(pred), " cols, expected 4 — check dropped factor level, fold ", k)
  
  actual_mat <- as.matrix(te[, .(Ch1, Ch2, Ch3, Ch4)])
  fold_losses[k] <- mlogloss(actual_mat, pred)
  oof_preds[train$fold == k, ] <- pred
}

cv_score <- mean(fold_losses)
cat("Fold losses:", fold_losses, "\n")
cat("nnet full-feature CV log-loss:", cv_score, "sd:", sd(fold_losses), "\n")

# --- Write OOF predictions for meta-learner use ---
oof_nnet <- data.table(Case = train$Case, Task = train$Task,
                       nnet_Ch1 = oof_preds[,1], nnet_Ch2 = oof_preds[,2],
                       nnet_Ch3 = oof_preds[,3], nnet_Ch4 = oof_preds[,4])
fwrite(oof_nnet, "oof_nnet.csv")
cat("Wrote oof_nnet.csv\n")