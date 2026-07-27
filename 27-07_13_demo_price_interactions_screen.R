# REQUIRES: rm(list=ls()), then source 27-07_1 and 27-07_10 only (verified 2026-07-27:
# 2-9 not needed, script 10 is self-contained). Confirm educ/region/gender exist
# as columns in train_long once sourced.

library(survival)
library(data.table)

# CV scorer = fold_loss_simple loop, parameterized on formula/score_formula.
score_clogit_cv <- function(fit_formula, score_formula, data_long, actual_full) {
  fold_loss <- numeric(5)
  for (i in 1:5) {
    tr_long_i <- data_long[fold != i]
    va_long_i <- copy(data_long[fold == i])
    
    fit <- tryCatch(
      clogit(fit_formula, data = tr_long_i, method = "exact"),
      error = function(e) { cat("Fold", i, "- clogit failed:", conditionMessage(e), "\n"); NULL }
    )
    if (is.null(fit)) { fold_loss[i] <- NA; next }
    if (any(is.na(coef(fit)))) {
      cat("Fold", i, "- WARNING: aliased coefficient(s):",
          paste(names(coef(fit))[is.na(coef(fit))], collapse = ", "), "\n")
    }
    
    mm <- model.matrix(score_formula, data = va_long_i)
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
    keep_coef <- names(coef(fit))[!is.na(coef(fit))]
    mm <- mm[, keep_coef, drop = FALSE]
    va_long_i[, score := as.numeric(mm %*% coef(fit)[keep_coef])]
    va_long_i[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]
    
    setorder(va_long_i, chid, alt)
    preds_wide <- dcast(va_long_i, chid ~ alt, value.var = "prob")
    setnames(preds_wide, as.character(1:4), c("Ch1","Ch2","Ch3","Ch4"))
    
    va_wide_order <- unique(va_long_i[, .(chid, Case, Task)])
    setkey(va_wide_order, chid)
    preds_wide <- merge(va_wide_order, preds_wide, by = "chid", sort = FALSE)
    
    actual_aligned <- merge(preds_wide[, .(Case, Task)], actual_full, by = c("Case","Task"), sort = FALSE)
    actual   <- as.matrix(actual_aligned[, .(Ch1, Ch2, Ch3, Ch4)])
    pred_mat <- as.matrix(preds_wide[, .(Ch1, Ch2, Ch3, Ch4)])
    
    fold_loss[i] <- mlogloss(actual, pred_mat)
    cat("Fold", i, "log-loss:", fold_loss[i], "\n")
  }
  fold_loss
}

actual_lookup <- train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]

# --- Step 0: verification (per Section 1 pipeline-refactor rule) ---
# Must reproduce Model 12 = 1.162971 (sd 0.01546) before trusting anything below.
verify_scores <- score_clogit_cv(simple_formula, simple_score_formula, train_long, actual_lookup)
cat(sprintf("\nVerification mean: %.6f | sd: %.6f (expect 1.162971 / 0.01546)\n\n",
            mean(verify_scores), sd(verify_scores)))

# educ/region/gender aren't in train_long yet (only segment/agea/Urbind were merged in).
# Same Case-level merge pattern as demo_lookup, just for these three instead.
extra_demo <- unique(train[, .(Case, educ, region, gender)])
train_long_ext <- merge(train_long, extra_demo, by = "Case", sort = FALSE)

# Fix factor levels on the full data so every fold's model.matrix() produces the same
# columns - otherwise a level missing from one fold's subset silently drops that column
# and keep_coef/mm indexing breaks ("subscript out of bounds").
train_long_ext[, educ   := factor(educ,   levels = sort(unique(educ)))]
train_long_ext[, region := factor(region, levels = sort(unique(region)))]
train_long_ext[, gender := factor(gender, levels = sort(unique(gender)))]

# --- Step 1: one-shot screen, educ/region/gender x Price added on top of Model 12 ---
model13_formula <- as.formula(paste0(
  "chosen ~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind",
  " + Price:educ + Price:region + Price:gender",
  " + strata(chid)"
))
model13_score_formula <- as.formula(paste0(
  "~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind",
  " + Price:educ + Price:region + Price:gender"
))

model13_scores <- score_clogit_cv(model13_formula, model13_score_formula, train_long_ext, actual_lookup)
cat(sprintf("\nModel 13 mean: %.6f | sd: %.6f\n", mean(model13_scores), sd(model13_scores)))

# Next (only if Model 13 beats 1.162971): split block to find which term(s) drive it, as Model 11->12.