# 28-07_12_oof_two_stage_and_blend.R  (v2 -- corrected after reading real source)
#
# Purpose: (a) generate out-of-fold predictions for Model 27 (two-stage
# opt-out model), keyed by Case+Task to match the real OOF file schema,
# and (b) re-run the ensemble blend-weight grid search with Model 27 as a
# 5th candidate alongside clogit15 / ranger18 / xgb20 / lc13.
#
# This is Next Steps Queue item #16 from experiment_log_2026-07-28_1605.md.
#
# CHANGES FROM v1 (corrected against actual source, not guessed):
#   - v1 wrongly assumed a `chid` column existed on wide `train` and used it
#     as the join key. It does not -- chid only exists on the long-format
#     data. The verified 28-07_11 cv_two_stage() joins on Case+Task, so this
#     version does too.
#   - v1 guessed OOF column names as {model}_Ch{1-4}. Real files use
#     clogit15_Ch1-4 / ranger18_Ch1-4 / xgb20_Ch1-4 / lc13_Ch1-4 (confirmed
#     from an actual `names()` printout), keyed on Case+Task with Ch1-4
#     already present as the ground-truth actual columns.
#   - The OOF-generation loop below is 28-07_11's own cv_two_stage(), kept
#     as close to verified-working as possible, just modified to also
#     collect predictions (not only the fold loss).
#
# PREREQS -- source in this exact order first (see their own header comments
# for why): 27-07_1 -> 27-07_10 -> 27-07_13 -> 28-07_7 -> 28-07_11
# ============================================================================

stopifnot(exists("train"), exists("stageA_formula"), exists("stageB_formula"),
          exists("stageB_score_formula"), exists("stageB_pool"),
          exists("actual_lookup"), exists("mlogloss"))

library(survival)
library(data.table)

# ----------------------------------------------------------------------------
# 1. OOF generation for Model 27 (mirrors 28-07_11's cv_two_stage(), extended
#    to also retain the fold's predictions, keyed on Case+Task)
# ----------------------------------------------------------------------------
oof_two_stage_generate <- function() {
  losses <- numeric(5)
  oof_list <- vector("list", 5)

  for (k in 1:5) {
    cat("Fold", k, "-- fitting Stage A/B on fold !=", k, "...\n")

    # ---- Stage A: P(opt-out), binary glm on the training folds ----
    fitA <- glm(stageA_formula, data = train[fold != k], family = binomial())
    pA   <- predict(fitA, newdata = train[fold == k], type = "response")  # P(Ch4=1)

    # ---- Stage B: clogit on Alts 1-3, fit on training folds, Ch4==0 only ----
    fitB <- clogit(stageB_formula, data = stageB_pool[fold != k & Ch4 == 0])
    teB  <- stageB_pool[fold == k]
    mm <- model.matrix(stageB_score_formula, data = teB)[, -1, drop = FALSE]

    bB <- coef(fitB)
    n_na <- sum(is.na(bB))
    if (n_na > 0) cat("  fold", k, "-- zeroing", n_na, "aliased NA coefficients (expected, see Section 2.2)\n")
    bB[is.na(bB)] <- 0

    teB[, linpred := as.vector(mm %*% bB)]
    # stable softmax over alts 1-3 within chid (subtract per-group max before exp)
    teB[, pB := { m <- max(linpred); ex <- exp(linpred - m); ex / sum(ex) }, by = chid]

    predB_wide <- dcast(teB, Case + Task ~ alt, value.var = "pB")
    setnames(predB_wide, c("1", "2", "3"), c("p1", "p2", "p3"))

    test_ids <- train[fold == k, .(Case, Task, Ch4)]
    test_ids[, pOptOut := pA]
    combined <- merge(test_ids, predB_wide, by = c("Case", "Task"))
    stopifnot(nrow(combined) == nrow(test_ids))  # catches a lossy merge silently dropping rows

    combined[, `:=`(
      predCh1 = (1 - pOptOut) * p1, predCh2 = (1 - pOptOut) * p2,
      predCh3 = (1 - pOptOut) * p3, predCh4 = pOptOut
    )]

    actual <- actual_lookup[combined, on = c("Case", "Task")]
    losses[k] <- mlogloss(as.matrix(actual[, .(Ch1, Ch2, Ch3, Ch4)]),
                           as.matrix(combined[, .(predCh1, predCh2, predCh3, predCh4)]))
    cat("Fold", k, "log-loss:", losses[k], "\n")

    oof_list[[k]] <- combined[, .(Case, Task,
                                   ts_Ch1 = predCh1, ts_Ch2 = predCh2,
                                   ts_Ch3 = predCh3, ts_Ch4 = predCh4)]
  }

  list(losses = losses, oof = rbindlist(oof_list))
}

result <- oof_two_stage_generate()
cat("\nModel 27 OOF CV -- mean:", mean(result$losses), " sd:", sd(result$losses), "\n")
cat("(compare to previously reported CV 1.163852, sd 0.01652 -- should match closely,\n")
cat(" this is the SAME fold-by-fold mean, not a pooled number, so it should be much\n")
cat(" closer than a pooled-vs-mean comparison would be)\n\n")

oof_two_stage_df <- result$oof
setorder(oof_two_stage_df, Case, Task)
stopifnot(nrow(oof_two_stage_df) == nrow(train))  # every row should have been scored exactly once
fwrite(oof_two_stage_df, "oof_two_stage.csv")

# ----------------------------------------------------------------------------
# 2. Blend-weight grid search, Model 27 as a 5th candidate
# ----------------------------------------------------------------------------
oof1 <- fread("oof_clogit_ranger_xgb.csv")
oof2 <- fread("oof_latent_class.csv")

cat("oof1 columns:", paste(names(oof1), collapse = ", "), "\n")
cat("oof2 columns:", paste(names(oof2), collapse = ", "), "\n")

stopifnot(all(c("Case", "Task") %in% names(oof1)), all(c("Case", "Task") %in% names(oof2)))

merged <- merge(oof1, oof2, by = c("Case", "Task"))
merged <- merge(merged, oof_two_stage_df, by = c("Case", "Task"))
stopifnot(nrow(merged) == nrow(oof1))  # catches any join silently dropping/duplicating rows

actual_mat <- as.matrix(merged[, .(Ch1, Ch2, Ch3, Ch4)])
clogit_mat <- as.matrix(merged[, .(clogit15_Ch1, clogit15_Ch2, clogit15_Ch3, clogit15_Ch4)])
ranger_mat <- as.matrix(merged[, .(ranger18_Ch1, ranger18_Ch2, ranger18_Ch3, ranger18_Ch4)])
xgb_mat    <- as.matrix(merged[, .(xgb20_Ch1,    xgb20_Ch2,    xgb20_Ch3,    xgb20_Ch4)])
lc_mat     <- as.matrix(merged[, .(lc13_Ch1,     lc13_Ch2,     lc13_Ch3,     lc13_Ch4)])
ts_mat     <- as.matrix(merged[, .(ts_Ch1,       ts_Ch2,       ts_Ch3,       ts_Ch4)])

grid_step <- 0.05  # matches the "finer grid" item already queued (#14)
w_seq <- seq(0, 1, grid_step)
weights_grid <- as.data.table(expand.grid(
  w_clogit = w_seq, w_ranger = w_seq, w_xgb = w_seq, w_lc = w_seq, w_ts = w_seq
))
weights_grid <- weights_grid[abs(w_clogit + w_ranger + w_xgb + w_lc + w_ts - 1) < 1e-8]
cat("Grid size after sum-to-1 filter:", nrow(weights_grid), "combinations\n")
cat("(this grid can be slow -- ~10k combinations x a matrix-weighted-sum over",
    nrow(merged), "rows each; if it's too slow, drop grid_step to 0.1 first as a quick pass)\n")

best <- list(loss = Inf, w = NULL)
for (i in seq_len(nrow(weights_grid))) {
  w <- weights_grid[i]
  blended <- w$w_clogit * clogit_mat + w$w_ranger * ranger_mat +
             w$w_xgb * xgb_mat + w$w_lc * lc_mat + w$w_ts * ts_mat
  loss <- mlogloss(actual_mat, blended)
  if (loss < best$loss) best <- list(loss = loss, w = w)
}

cat("\nBest 5-way blend log-loss:", best$loss, "\n")
cat("Prior 4-way best (Model 21):  1.156089\n")
print(best$w)

# If best$loss < 1.156089: Model 27 earns ensemble-diversity weight -- becomes
# Model 28 in the Registry, update Section 1/3/6/7 accordingly.
# If not: Model 27 is fully closed (standalone AND ensemble), update
# Section 5 "Things Tried and Ruled Out" and remove item #16 from the queue.

