# =============================================================================
# Model B' — clogit baseline (current best: CV 1.180861, LB 1.254)
#
# Replaces day1_2_bp.R: identical model/logic, just requires
# 27-07_1_pipeline_foundation.R instead of the full day1_2.R (no mlogit fits,
# no diagnostics needed to get here — see that script's header for why).
#
# Run 27-07_1_pipeline_foundation.R first, then this.
#
# THIS IS THE BASE FOR DAY 3 TRACK A (demographic x attribute interactions):
# extend `clogit_formula` below by adding interaction terms, e.g.
#   ... + CC:segment + Price:incomea  (etc.)
# Keep everything else (fold loop, scoring-by-hand, mlogloss call) unchanged
# so CV numbers stay comparable to this baseline.
# =============================================================================

library(survival)

clogit_formula <- as.formula(
  paste0("chosen ~ ", paste(c(identified_vars, "Price"), collapse = " + "), " + strata(chid)")
)
score_formula <- as.formula(paste("~", paste(c(identified_vars, "Price"), collapse = " + ")))

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

  # predict.coxph fails on new strata levels (every validation chid is new,
  # by design of Case-grouped CV). Strata are conditioned out of the
  # conditional-logit likelihood, so score directly from fitted coefficients
  # instead of calling predict().
  mm <- model.matrix(score_formula, data = va_long_i)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm <- mm[, names(coef(fit_Bp)), drop = FALSE]
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
cat("\nExpect ~1.180861 mean if reproduced exactly — if not, flag before building on top.\n")
