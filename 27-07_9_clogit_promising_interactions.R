# 27-07_9_clogit_promising_interactions.R
# Full 5-fold CV on the 3 interaction terms that cleared BOTH significance and
# magnitude bars in 27-07_8_interaction_screen.R:
#   Price:agea_z    (coef -0.166, z=-16.3, ~12% of Price's main effect)
#   Price:segment   (2 of 4 non-ref levels large+significant: Prestige Luxury
#                    Sedan +0.550 z=11.4, Midsize Luxury Utility +0.536 z=10.6)
#   Price:Urbind    (coef +0.107, z=7.8, ~8% of Price's main effect)
# Excluded: Price:milesa_z and Price:nighta_z — milesa_z was statistically
# significant but tiny (~2.8% of main effect, same pattern as the ruled-out
# incomea interaction); nighta_z was not significant at all.
#
# Run 27-07_1_pipeline_foundation.R then 27-07_8_interaction_screen.R first
# (needs train_long already merged with segment/agea_z/Urbind).

library(survival)

rhs_vars <- c(identified_vars, "Price")
promising_formula <- as.formula(
  paste0("chosen ~ ", paste(rhs_vars, collapse = " + "),
         " + Price:agea_z + Price:segment + Price:Urbind + strata(chid)")
)
promising_score_formula <- as.formula(
  paste("~", paste(rhs_vars, collapse = " + "),
        "+ Price:agea_z + Price:segment + Price:Urbind")
)

fold_loss_promising <- numeric(5)
for (i in 1:5) {
  tr_long_i <- train_long[fold != i]
  va_long_i <- copy(train_long[fold == i])
  
  fit_promising <- tryCatch(
    clogit(promising_formula, data = tr_long_i, method = "exact"),
    error = function(e) {
      cat("Fold", i, "- clogit failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(fit_promising)) { fold_loss_promising[i] <- NA; next }
  
  if (any(is.na(coef(fit_promising)))) {
    cat("Fold", i, "- WARNING: aliased coefficient(s):",
        paste(names(coef(fit_promising))[is.na(coef(fit_promising))], collapse = ", "), "\n")
  }
  
  mm <- model.matrix(promising_score_formula, data = va_long_i)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  keep_coef <- names(coef(fit_promising))[!is.na(coef(fit_promising))]
  mm <- mm[, keep_coef, drop = FALSE]
  va_long_i[, score := as.numeric(mm %*% coef(fit_promising)[keep_coef])]
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
  
  fold_loss_promising[i] <- mlogloss(actual, pred_mat)
  cat("clogit+3interactions fold", i, "log-loss:", fold_loss_promising[i], "\n")
}

cat("\nclogit + Price:{agea_z, segment, Urbind} — CV summary\n")
cat("Mean:", mean(fold_loss_promising, na.rm = TRUE), " SD:", sd(fold_loss_promising, na.rm = TRUE),
    "(", sum(is.na(fold_loss_promising)), "fold(s) skipped)\n")
cat("Per-fold:", paste(round(fold_loss_promising, 5), collapse = ", "), "\n")
cat("vs baseline (no interactions): 1.180861\n")