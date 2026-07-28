# 29-07_1_reverify_model29.R
# REQUIRES FIRST (same session, in order): 27-07_1, 27-07_10, 27-07_12/13, 28-07_7,
#   28-07_11 (defines stageA_formula/stageB_formula/stageB_score_formula, merges
#   is_male/is_luxury_segment/agea_z/Ch4 onto train/train_long_ext),
#   then 28-07_5 UNMODIFIED (defines train2, train_long_ext2, fold_map,
#   fold2_rowaligned, oof_clogit, oof_ranger, oof_xgb, actual_mat, mlogloss()).
# Do not skip 28-07_5 - this script reuses its objects rather than rebuilding them.
#
# Purpose: extend the seed-42 reverification to the two-stage opt-out model (Model 27)
# and rerun the blend as a 4-way simplex (clogit15/ranger18/xgb20/two-stage27),
# comparable to Model 29 (lc13 excluded - weight 0 in Model 29 already).

library(data.table); library(survival)

stopifnot(exists("oof_clogit"), exists("oof_ranger"), exists("oof_xgb"),
          exists("train2"), exists("fold_map"), exists("train_long_ext2"))

# ---- two-stage OOF, fold2 (Case-grouped, same split as clogit/ranger/xgb above) ----
train_long_ext2[, fold2 := fold_map$fold2[match(Case, fold_map$Case)]]
stageB_pool2 <- train_long_ext2[alt %in% 1:3]

n <- nrow(train2)
oof_ts  <- matrix(NA_real_, n, 4, dimnames = list(NULL, c("Ch1","Ch2","Ch3","Ch4")))
oof_key <- train2[, .(Case, Task)]

for (f in 1:5) {
  fitA <- glm(stageA_formula, data = train2[fold2 != f], family = binomial())
  pA   <- predict(fitA, newdata = train2[fold2 == f], type = "response")
  
  fitB <- clogit(stageB_formula, data = stageB_pool2[fold2 != f & Ch4 == 0])
  teB  <- copy(stageB_pool2[fold2 == f])
  mm <- model.matrix(stageB_score_formula, data = teB)[, -1, drop = FALSE]
  
  bB <- coef(fitB)
  bB[is.na(bB)] <- 0   # aliased coefs -> 0, same fix as 28-07_11
  teB[, linpred := as.vector(mm %*% bB)]
  teB[, expu := exp(linpred)]
  teB[, pB := expu / sum(expu), by = chid]
  
  predB_wide <- dcast(teB, Case + Task ~ alt, value.var = "pB")
  setnames(predB_wide, c("1","2","3"), c("p1","p2","p3"))
  
  test_ids <- train2[fold2 == f, .(Case, Task, Ch4)]
  test_ids[, pOptOut := pA]
  combined <- merge(test_ids, predB_wide, by = c("Case","Task"))
  combined[, `:=`(
    predCh1 = (1 - pOptOut) * p1, predCh2 = (1 - pOptOut) * p2,
    predCh3 = (1 - pOptOut) * p3, predCh4 = pOptOut
  )]
  
  row_idx <- match(paste(combined$Case, combined$Task), paste(oof_key$Case, oof_key$Task))
  oof_ts[row_idx, ] <- as.matrix(combined[, .(predCh1, predCh2, predCh3, predCh4)])
}
stopifnot(!anyNA(oof_ts))

cat("two-stage27 fresh-seed (42) CV:", mlogloss(actual_mat, oof_ts), " (orig 1.163852)\n")

# ---- 4-way blend grid, step 0.05 (matches Model 29's grid resolution) ----
step <- 0.05; ws <- seq(0, 1, by = step)
grid <- expand.grid(w_clogit = ws, w_ranger = ws, w_xgb = ws, w_ts = ws)
grid <- grid[abs(rowSums(grid) - 1) < 1e-8, ]
grid$loss <- NA_real_
for (i in seq_len(nrow(grid))) {
  blend <- grid$w_clogit[i]*oof_clogit + grid$w_ranger[i]*oof_ranger +
    grid$w_xgb[i]*oof_xgb + grid$w_ts[i]*oof_ts
  grid$loss[i] <- mlogloss(actual_mat, blend)
}
grid <- grid[order(grid$loss), ]
cat("\nTop 10 fresh-seed (42) 4-way blends:\n"); print(head(grid, 10))

cat("\nOriginal (seed 2024) Model 29: 1.154955 @ w_clogit=0.4 w_ranger=0.1 w_xgb=0.2 w_ts=0.3\n")
cat("Original (seed 2024) Model 21: 1.156089 @ w_clogit=0.6 w_ranger=0.1 w_xgb=0.3 w_ts=0\n")
cat("Model 21's own known reseeding shift: 0.000870 (this is the noise bar to beat)\n")