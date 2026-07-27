# 28-07_3_ensemble_blend.R
# REQUIRES: run 28-07_1 and 28-07_2 first (each writes a CSV to the working
# directory: oof_clogit_ranger_xgb.csv, oof_latent_class.csv). Otherwise
# fully self-contained — a fresh R session is fine, no upstream objects needed.
#
# Purpose: merge all four models' OOF predictions, check the latent-class /
# clogit correlation (open question, log Section 6), and grid-search blend
# weights to minimize CV log-loss.

library(data.table)

mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

oof_a <- fread("oof_clogit_ranger_xgb.csv")
oof_b <- fread("oof_latent_class.csv")
oof <- merge(oof_a, oof_b, by = c("Case","Task"), sort = FALSE)
stopifnot(nrow(oof) == nrow(oof_a))  # confirm the merge didn't drop/duplicate rows

actual_mat <- as.matrix(oof[, .(Ch1, Ch2, Ch3, Ch4)])
m_clogit <- as.matrix(oof[, .(clogit15_Ch1, clogit15_Ch2, clogit15_Ch3, clogit15_Ch4)])
m_ranger <- as.matrix(oof[, .(ranger18_Ch1, ranger18_Ch2, ranger18_Ch3, ranger18_Ch4)])
m_xgb    <- as.matrix(oof[, .(xgb20_Ch1, xgb20_Ch2, xgb20_Ch3, xgb20_Ch4)])
m_lc     <- as.matrix(oof[, .(lc13_Ch1, lc13_Ch2, lc13_Ch3, lc13_Ch4)])

cat("--- single-model OOF check (should match logged CV) ---\n")
cat("clogit15:", mlogloss(actual_mat, m_clogit), "\n")
cat("ranger18:", mlogloss(actual_mat, m_ranger), "\n")
cat("xgb20:   ", mlogloss(actual_mat, m_xgb), "\n")
cat("lc13:    ", mlogloss(actual_mat, m_lc), "\n")

# ---- open question: latent-class / clogit correlation, on P(chosen alt) ----
chosen_idx <- max.col(actual_mat)
p_lc     <- m_lc[cbind(seq_len(nrow(m_lc)), chosen_idx)]
p_clogit <- m_clogit[cbind(seq_len(nrow(m_clogit)), chosen_idx)]
cat("\ncor(lc13, clogit15) on P(chosen alt):", cor(p_lc, p_clogit), "\n")

# ---- blend weight grid search (simplex, step 0.1, 4 models) ----
step <- 0.1
ws <- seq(0, 1, by = step)
grid <- expand.grid(w_clogit = ws, w_ranger = ws, w_xgb = ws, w_lc = ws)
grid <- grid[abs(rowSums(grid) - 1) < 1e-8, ]

grid$loss <- NA_real_
for (i in seq_len(nrow(grid))) {
  blend <- grid$w_clogit[i]*m_clogit + grid$w_ranger[i]*m_ranger +
    grid$w_xgb[i]*m_xgb + grid$w_lc[i]*m_lc
  grid$loss[i] <- mlogloss(actual_mat, blend)
}

grid <- grid[order(grid$loss), ]
cat("\nTop 10 blends:\n")
print(head(grid, 10))

fwrite(grid, "ensemble_blend_grid.csv")
cat("\nSaved ensemble_blend_grid.csv (", nrow(grid), "combos)\n")