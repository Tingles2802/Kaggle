# 27-07_3_clogit_interactions.R
# Extends confirmed baseline (27-07_2_clogit_baseline.R, CV 1.180861) with
# Price x incomea interaction. FIXED: incomea now standardized (raw scale
# 9,000-3,900,000 was causing a scale-mismatch artifact — see log Section 5).
# Idempotent: safe to re-source without re-running pipeline_foundation.R first.

library(survival)

demo_cols <- c("Case", "incomea")
stopifnot(all(demo_cols %in% names(train)))

# Idempotency guard: drop previously-merged demo/interaction cols if present
for (col in c("incomea", "incomea_z", "Price_income")) {
  if (col %in% names(train_long)) train_long[, (col) := NULL]
}

demo_lookup <- unique(train[, ..demo_cols])
stopifnot(nrow(demo_lookup) == uniqueN(train_long$Case))

train_long <- merge(train_long, demo_lookup, by = "Case", sort = FALSE)

# FIX: standardize incomea before interacting with standardized Price
train_long[, incomea_z := as.numeric(scale(incomea))]
train_long[, Price_income := Price * incomea_z]

interact_formula <- as.formula(
  paste0("chosen ~ ", paste(c(identified_vars, "Price", "Price_income"), collapse = " + "),
         " + strata(chid)")
)
interact_score_formula <- as.formula(
  paste("~", paste(c(identified_vars, "Price", "Price_income"), collapse = " + "))
)

fold_loss_interact <- numeric(5)
for (i in 1:5) {
  tr_long_i <- train_long[fold != i]
  va_long_i <- copy(train_long[fold == i])
  
  fit_interact <- tryCatch(
    clogit(interact_formula, data = tr_long_i, method = "exact"),
    error = function(e) {
      cat("Fold", i, "- clogit failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(fit_interact)) { fold_loss_interact[i] <- NA; next }
  
  mm <- model.matrix(interact_score_formula, data = va_long_i)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm <- mm[, names(coef(fit_interact)), drop = FALSE]
  va_long_i[, score := as.numeric(mm %*% coef(fit_interact))]
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
  
  fold_loss_interact[i] <- mlogloss(actual, pred_mat)
  cat("Interaction model fold", i, "log-loss:", fold_loss_interact[i], "\n")
}

cat("\nModel: clogit + Price_income (incomea standardized) — CV summary\n")
cat("Mean:", mean(fold_loss_interact, na.rm = TRUE), " SD:", sd(fold_loss_interact, na.rm = TRUE), "\n")
cat("vs baseline (fit_Bp):", mean(fold_loss_Bp, na.rm = TRUE), "\n")
print(summary(fit_interact)$coefficients["Price_income", ])  # sanity-check coef scale/significance