# 27-07_7_clogit_asc4.R
# Step 2 of the alt-4 investigation (see experiment_log Section 2, Gotcha #4 /
# Open Questions, and 27-07_6_alt4_check.R's confirmed finding: alt-4 is a fixed
# profile — all 19 attrs + Price constant at 0, in both train and test — and it's
# the MOST chosen alternative, 30.2% share).
#
# Rationale: the current no-ASC clogit baseline (Model Registry #5, CV 1.180861)
# already gives alt-4 a fixed utility, but only as a byproduct of each attribute's
# "level 0 / inactive" coefficient (shared with partial-profile inactivity on
# alts 1-3) plus Price's coefficient at scaled(0). That's not the same as letting
# alt-4 have its own freely-estimated baseline appeal. This script adds a single
# `asc4` dummy (I(alt==4)) to test that directly. Only one ASC (not 3) because
# alts 1-3 are structurally symmetric/randomized in the design; only alt-4 is
# uniquely fixed.
#
# Run 27-07_1_pipeline_foundation.R first, then this (does NOT need 27-07_2 —
# self-contained, but mirrors its scoring logic exactly for comparability).

library(survival)

# ---- Add the ASC column ----
train_long[, asc4 := as.numeric(alt == 4)]

# ---- Model spec: baseline + asc4 ----
rhs_vars <- c(identified_vars, "Price", "asc4")
clogit_formula_asc4 <- as.formula(
  paste0("chosen ~ ", paste(rhs_vars, collapse = " + "), " + strata(chid)")
)
score_formula_asc4 <- as.formula(paste("~", paste(rhs_vars, collapse = " + ")))

fold_loss_asc4 <- numeric(5)
for (i in 1:5) {
  tr_long_i <- train_long[fold != i]
  va_long_i <- copy(train_long[fold == i])
  
  fit_asc4 <- tryCatch(
    clogit(clogit_formula_asc4, data = tr_long_i, method = "exact"),
    error = function(e) {
      cat("Fold", i, "- clogit failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  
  if (is.null(fit_asc4)) {
    fold_loss_asc4[i] <- NA
    next
  }
  
  # Flag if asc4 (or anything else) came out NA/aliased — would indicate the
  # quasi-collinearity concern (asc4 vs. the 19 attributes' shared level-0
  # coefficients) is a real identification problem, not just a theoretical one.
  if (any(is.na(coef(fit_asc4)))) {
    cat("Fold", i, "- WARNING: aliased coefficient(s):",
        paste(names(coef(fit_asc4))[is.na(coef(fit_asc4))], collapse = ", "), "\n")
  }
  
  mm <- model.matrix(score_formula_asc4, data = va_long_i)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  keep_coef <- names(coef(fit_asc4))[!is.na(coef(fit_asc4))]
  mm <- mm[, keep_coef, drop = FALSE]
  va_long_i[, score := as.numeric(mm %*% coef(fit_asc4)[keep_coef])]
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
  
  fold_loss_asc4[i] <- mlogloss(actual, pred_mat)
  cat("clogit+asc4 fold", i, "log-loss:", fold_loss_asc4[i], "\n")
}

cat("\nclogit + asc4 (single alt-4 ASC) — CV summary\n")
cat("Mean:", mean(fold_loss_asc4, na.rm = TRUE), " SD:", sd(fold_loss_asc4, na.rm = TRUE),
    "(", sum(is.na(fold_loss_asc4)), "fold(s) skipped)\n")
cat("Per-fold:", paste(round(fold_loss_asc4, 5), collapse = ", "), "\n")
cat("vs baseline (no ASC): 1.180861\n")

# ---- Report the asc4 coefficient itself (full-data fit, for interpretation) ----
fit_full <- clogit(clogit_formula_asc4, data = train_long, method = "exact")
cat("\nFull-data asc4 coefficient (log-odds of alt-4 vs. attribute-driven alts, controlling for Price/attrs):\n")
print(summary(fit_full)$coefficients["asc4", ])