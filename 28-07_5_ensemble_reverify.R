# 28-07_5_ensemble_reverify.R
# REQUIRES FIRST (in order): 27-07_1 (foundation), 27-07_4 Steps 1-2 (ranger_data/X_cols/fold),
#   27-07_5 (xgb_data_oh/numeric_cols/y_zero_indexed), 27-07_10, 27-07_13 (train_long_ext,
#   rhs_vars, score_clogit_cv(), actual_lookup). Fresh session fine as long as those ran.
#
# Purpose: refit clogit15/ranger18/xgb20 under a NEW Case-level fold seed (42, not 2024)
# and rerun the blend grid search, to check whether Model 21's weights/CV (1.156089) are
# robust to fold-split choice (Section 6 open question). lc13 excluded - already ruled out,
# zero weight in every top-10 blend of the original search.
#
# ASSUMPTIONS TO VERIFY (not in the 5 files I was given):
#   (a) Model 15 formula = Model 12's rhs_vars/interactions + Price:gender. Step 0 below
#       checks this against the logged CV (1.162469) on the ORIGINAL fold - if it doesn't
#       match, stop and paste 27-07_14/15.
#   (b) ranger_data has NO Case column (confirmed by error on first run) - it is row-aligned
#       with train directly (per 27-07_17's pattern: ranger_data[fold!=f] paired positionally
#       with train[fold==f,...], not joined by key). fold2 below is applied by row position,
#       same as the xgboost matrices. If ranger_data's row order ever differs from train's
#       (e.g. any sort/filter during its construction in 27-07_4), this will silently misalign -
#       paste 27-07_4 if fresh-seed ranger numbers look implausible.

library(survival); library(ranger); library(xgboost); library(data.table)

mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

# ---- Step 0: verify inferred Model 15 formula on the ORIGINAL fold (pipeline-refactor rule) ----
model15_formula <- as.formula(paste0(
  "chosen ~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender + strata(chid)"
))
model15_score_formula <- as.formula(paste0(
  "~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender"
))

verify15 <- score_clogit_cv(model15_formula, model15_score_formula, train_long_ext, actual_lookup)
cat(sprintf("Step 0 verify Model 15: mean %.6f  sd %.6f  (expect 1.162469 / 0.015443)\n",
            mean(verify15), sd(verify15)))
stopifnot(abs(mean(verify15) - 1.162469) < 1e-4)  # STOP HERE if mismatch - formula is wrong

# ---- Fresh Case-level fold assignment (seed 42) ----
set.seed(42)
case_ids <- unique(train$Case)
fold_map <- data.table(Case = case_ids, fold2 = sample(rep(1:5, length.out = length(case_ids))))

train2          <- merge(train, fold_map, by = "Case", sort = FALSE)
stopifnot(nrow(train2) == nrow(train))                       # order must stay aligned to xgb/ranger rows
train_long_ext2 <- merge(train_long_ext[, !"fold", with = FALSE], fold_map, by = "Case", sort = FALSE)

# fold2 by row POSITION, aligned to train's original row order - used for both ranger_data
# and xgb_data_oh (neither carries a Case column; both are row-aligned to train, see assumption b)
fold2_rowaligned <- fold_map$fold2[match(train$Case, fold_map$Case)]
stopifnot(length(fold2_rowaligned) == nrow(train), nrow(ranger_data) == nrow(train))

ranger_data2 <- copy(ranger_data)
ranger_data2[, fold2 := fold2_rowaligned]
fold2_xgb <- fold2_rowaligned

n <- nrow(train)
oof_clogit <- matrix(NA_real_, n, 4, dimnames = list(NULL, c("Ch1","Ch2","Ch3","Ch4")))
oof_ranger <- matrix(NA_real_, n, 4, dimnames = list(NULL, c("Ch1","Ch2","Ch3","Ch4")))
oof_xgb    <- matrix(NA_real_, n, 4, dimnames = list(NULL, c("Ch1","Ch2","Ch3","Ch4")))
oof_key    <- train2[, .(Case, Task)]

# ---- clogit OOF, fold2 ----
for (f in 1:5) {
  tr <- train_long_ext2[fold2 != f]
  va <- copy(train_long_ext2[fold2 == f])
  fit <- clogit(model15_formula, data = tr, method = "exact")
  mm <- model.matrix(model15_score_formula, data = va)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  keep_coef <- names(coef(fit))[!is.na(coef(fit))]
  mm <- mm[, keep_coef, drop = FALSE]
  va[, score := as.numeric(mm %*% coef(fit)[keep_coef])]
  va[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]
  setorder(va, chid, alt)
  pw <- dcast(va, chid ~ alt, value.var = "prob")
  setnames(pw, as.character(1:4), c("Ch1","Ch2","Ch3","Ch4"))
  key_i <- unique(va[, .(chid, Case, Task)]); setkey(key_i, chid)
  pw <- merge(key_i, pw, by = "chid", sort = FALSE)
  row_idx <- match(paste(pw$Case, pw$Task), paste(oof_key$Case, oof_key$Task))
  oof_clogit[row_idx, ] <- as.matrix(pw[, .(Ch1, Ch2, Ch3, Ch4)])
}

# ---- ranger OOF, fold2 (row-position aligned to train, same as xgb below) ----
for (f in 1:5) {
  tr_idx <- which(fold2_rowaligned != f)
  va_idx <- which(fold2_rowaligned == f)
  tr <- ranger_data2[tr_idx]
  va <- ranger_data2[va_idx]
  fit <- ranger(y ~ ., data = tr[, !c("fold2"), with = FALSE], probability = TRUE,
                num.trees = 500, mtry = 40, min.node.size = 60, seed = 2024)
  preds <- predict(fit, data = va[, !c("fold2"), with = FALSE])$predictions
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  oof_ranger[va_idx, ] <- preds
}

# ---- xgboost OOF, fold2 ----
for (f in 1:5) {
  tr_idx <- which(fold2_xgb != f)
  va_idx <- which(fold2_xgb == f)
  dtrain <- xgb.DMatrix(data = xgb_data_oh[tr_idx, ], label = y_zero_indexed[tr_idx])
  dtest  <- xgb.DMatrix(data = xgb_data_oh[va_idx, ], label = y_zero_indexed[va_idx])
  fit <- xgb.train(params = list(objective = "multi:softprob", num_class = 4,
                                 eval_metric = "mlogloss", eta = 0.1, max_depth = 3),
                   data = dtrain, nrounds = 1000, evals = list(val = dtest),
                   early_stopping_rounds = 20, verbose = 0)
  best_iter <- as.integer(xgb.attr(fit, "best_iteration")) + 1L
  preds <- predict(fit, dtest, iterationrange = c(1L, best_iter))
  colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
  oof_xgb[va_idx, ] <- preds
}

# ---- sanity: fresh-seed single-model CV vs originals ----
actual_mat <- as.matrix(train2[, .(Ch1, Ch2, Ch3, Ch4)])
cat("\nFresh-seed (42) single-model CV vs original (seed 2024):\n")
cat("clogit15:", mlogloss(actual_mat, oof_clogit), " (orig 1.162469)\n")
cat("ranger18:", mlogloss(actual_mat, oof_ranger), " (orig 1.189566)\n")
cat("xgb20:   ", mlogloss(actual_mat, oof_xgb),    " (orig 1.176378)\n")

# ---- blend grid search, 3-model simplex (lc13 excluded - ruled out) ----
step <- 0.1; ws <- seq(0, 1, by = step)
grid <- expand.grid(w_clogit = ws, w_ranger = ws, w_xgb = ws)
grid <- grid[abs(rowSums(grid) - 1) < 1e-8, ]
grid$loss <- NA_real_
for (i in seq_len(nrow(grid))) {
  blend <- grid$w_clogit[i]*oof_clogit + grid$w_ranger[i]*oof_ranger + grid$w_xgb[i]*oof_xgb
  grid$loss[i] <- mlogloss(actual_mat, blend)
}
grid <- grid[order(grid$loss), ]
cat("\nTop 10 blends (seed 42):\n"); print(head(grid, 10))
cat("\nOriginal (seed 2024) Model 21: 1.156089 at clogit 0.6 / ranger 0.1 / xgb 0.3\n")