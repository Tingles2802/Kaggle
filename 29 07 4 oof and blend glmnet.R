# 29-07_4_oof_and_blend_glmnet.R
# REQUIRES (same R session, no refit needed): 29-07_3_glmnet_multinomial.R
#   already run (needs oof_preds, train, mlogloss in memory).
# ALSO REQUIRES on disk (already generated in prior sessions):
#   oof_clogit_ranger_xgb.csv (28-07_1), oof_two_stage.csv (28-07_12).
#   If either is missing, regenerate via those scripts first — don't guess
#   values.
#
# Purpose: (1) save+verify glmnet's OOF predictions (Model 30 candidate),
# (2) test whether adding it to the ensemble beats Model 29 (1.154955).
# lc13 excluded from the grid — confirmed weight-0 in every blend tried.

library(data.table)

# ---- 1. Save + verify glmnet OOF ----
oof_glmnet <- data.table(Case = train$Case, Task = train$Task, oof_preds)
setnames(oof_glmnet, c("V1","V2","V3","V4"),
         c("glmnet30_Ch1","glmnet30_Ch2","glmnet30_Ch3","glmnet30_Ch4"))
stopifnot(!anyNA(oof_glmnet))

actual_mat <- as.matrix(train[, .(Ch1, Ch2, Ch3, Ch4)])
check_score <- mlogloss(actual_mat, as.matrix(oof_glmnet[, .(glmnet30_Ch1,glmnet30_Ch2,glmnet30_Ch3,glmnet30_Ch4)]))
cat("glmnet30 OOF reproduces CV (expect ~1.182182):", check_score, "\n")
stopifnot(abs(check_score - 1.182182) < 1e-4)

fwrite(oof_glmnet, "oof_glmnet.csv")
cat("Saved oof_glmnet.csv (", nrow(oof_glmnet), "rows)\n")

# ---- 2. Merge with existing ensemble OOF files ----
oof_ctx <- fread("oof_clogit_ranger_xgb.csv")   # Case,Task,Ch1-4,clogit15_*,ranger18_*,xgb20_*
oof_ts  <- fread("oof_two_stage.csv")           # Case,Task,ts_Ch1-4

merged <- merge(oof_ctx, oof_ts, by = c("Case","Task"), sort = FALSE)
merged <- merge(merged, oof_glmnet, by = c("Case","Task"), sort = FALSE)
stopifnot(nrow(merged) == nrow(train))

actual_mat <- as.matrix(merged[, .(Ch1, Ch2, Ch3, Ch4)])
P_clogit <- as.matrix(merged[, .(clogit15_Ch1,clogit15_Ch2,clogit15_Ch3,clogit15_Ch4)])
P_ranger <- as.matrix(merged[, .(ranger18_Ch1,ranger18_Ch2,ranger18_Ch3,ranger18_Ch4)])
P_xgb    <- as.matrix(merged[, .(xgb20_Ch1,xgb20_Ch2,xgb20_Ch3,xgb20_Ch4)])
P_ts     <- as.matrix(merged[, .(ts_Ch1,ts_Ch2,ts_Ch3,ts_Ch4)])
P_glmnet <- as.matrix(merged[, .(glmnet30_Ch1,glmnet30_Ch2,glmnet30_Ch3,glmnet30_Ch4)])

# ---- 3. 5-way blend-weight grid (0.05 step, sum-to-1) ----
step <- 0.05
n_steps <- round(1 / step)
grid_raw <- expand.grid(w1 = 0:n_steps, w2 = 0:n_steps, w3 = 0:n_steps, w4 = 0:n_steps)
grid_raw <- grid_raw[rowSums(grid_raw) <= n_steps, ]
grid_raw$w5 <- n_steps - rowSums(grid_raw)
weights_grid <- grid_raw * step
cat("Grid size:", nrow(weights_grid), "combos\n")   # expect 10,626, matching 28-07_12's precedent

best_loss <- Inf
best_w <- NULL
for (i in seq_len(nrow(weights_grid))) {
  w <- as.numeric(weights_grid[i, ])
  blend <- w[1]*P_clogit + w[2]*P_ranger + w[3]*P_xgb + w[4]*P_ts + w[5]*P_glmnet
  loss <- mlogloss(actual_mat, blend)
  if (loss < best_loss) { best_loss <- loss; best_w <- w }
}

cat("\nBest 5-way blend (incl. glmnet30) — CV:", best_loss, "\n")
cat("Weights (clogit15, ranger18, xgb20, two-stage27, glmnet30):",
    paste(round(best_w, 3), collapse = ", "), "\n")
cat("\nvs Model 29 (current best, no glmnet):", 1.154955, "\n")
cat("Improvement:", 1.154955 - best_loss, "\n")