# =============================================================================
# Model B' — conditional logit via survival::clogit (in base R, no extra
# install needed)
#
# WHY: three full diagnostic rounds (row-level rank 61/61, within-chid rank
# 61/61 on full data, within-chid rank 61/61 on EVERY fold's training
# subset, full 19-attr+Price model fits fine on full data, leave-one-out all
# OK) have ruled out every design/collinearity/identification explanation
# for mlogit's persistent "system is computationally singular" /
# "exactly singular: U[4,4] = 0" error. That combination (full rank
# everywhere, still singular at the actual fit) points to numerical
# instability in mlogit's own optimizer -- most likely quasi-complete
# separation (some sparse attribute-level combination that near-perfectly
# predicts the chosen alternative within a fold), not a data problem.
# clogit fits the identical statistical model (conditional/McFadden logit,
# each chid as a matched stratum) via a Cox-based optimizer that is
# substantially more robust to this exact failure mode.
#
# LIMITATION: like mlogit's "| 0" case, strata(chid) absorbs any
# alternative-specific constants -- clogit cannot estimate ASCs, same as
# the no-ASC retry already being used. The attribute + Price coefficients
# (the actual object of interest) are unaffected by this.
# =============================================================================

library(survival)

# Reuses train_long, case_folds, cat_attr_vars, identified_vars, mlogloss()
# already built/defined in day2_models_multinom_mlogit.R -- run that first.

clogit_formula <- as.formula(
  paste0("chosen ~ ", paste(identified_vars, collapse = " + "), " + strata(chid)")
)

fold_loss_Bp <- numeric(5)
for (i in 1:5) {
  tr_long_i <- train_long[fold != i]
  va_long_i <- copy(train_long[fold == i])
  
  fit_Bp <- tryCatch(
    clogit(clogit_formula, data = tr_long_i, method = "exact"),
    error = function(e) {
      cat("Fold", i, "- clogit failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  
  if (is.null(fit_Bp)) {
    fold_loss_Bp[i] <- NA
    next
  }
  
  # FIX: predict.coxph tries to match strata(chid) factor levels between
  # train and newdata. Since Case-grouped folds mean validation chids never
  # appear in training (0 overlap by design), EVERY validation chid is a
  # "new level" and predict() fails outright. Strata are conditioned out of
  # the conditional-logit likelihood and don't enter the linear score at
  # all, so bypass predict() entirely and compute the score directly from
  # the fitted attribute/Price coefficients.
  score_formula <- as.formula(paste("~", paste(identified_vars, collapse = " + ")))
  mm <- model.matrix(score_formula, data = va_long_i)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm <- mm[, names(coef(fit_Bp)), drop = FALSE]  # align to fitted coefficient names/order
  va_long_i[, score := as.numeric(mm %*% coef(fit_Bp))]
  va_long_i[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]
  
  setorder(va_long_i, chid, alt)
  preds_wide <- dcast(va_long_i, chid ~ alt, value.var = "prob")
  setnames(preds_wide, as.character(1:4), c("Ch1","Ch2","Ch3","Ch4"))
  
  va_wide_order <- unique(va_long_i[, .(chid, Case, Task)])
  setkey(va_wide_order, chid)
  preds_wide <- merge(va_wide_order, preds_wide, by = "chid", sort = FALSE)
  
  actual_lookup <- train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]
  actual_aligned <- merge(preds_wide[, .(Case, Task)], actual_lookup, by = c("Case","Task"), sort = FALSE)
  
  actual <- as.matrix(actual_aligned[, .(Ch1, Ch2, Ch3, Ch4)])
  pred_mat <- as.matrix(preds_wide[, .(Ch1, Ch2, Ch3, Ch4)])
  
  fold_loss_Bp[i] <- mlogloss(actual, pred_mat)
  cat("Model B' (clogit) fold", i, "log-loss:", fold_loss_Bp[i], "\n")
}

cat("\nModel B' (clogit, alt-specific attributes, no ASCs) — CV summary\n")
cat("Mean:", mean(fold_loss_Bp, na.rm = TRUE), " SD:", sd(fold_loss_Bp, na.rm = TRUE),
    "(", sum(is.na(fold_loss_Bp)), "fold(s) skipped)\n")
cat("Per-fold:", paste(round(fold_loss_Bp, 5), collapse = ", "), "\n")