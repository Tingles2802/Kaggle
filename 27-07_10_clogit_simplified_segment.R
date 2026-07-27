# 27-07_10_clogit_simplified_segment.R
# Tests a simplified `is_luxury_segment` binary interaction in place of the full
# `segment` categorical interaction used in Model Registry #11 (CV 1.164339).
# Rationale: in screening (27-07_8), only 2 of 4 non-reference segment levels
# (both containing "Luxury" in the label) were individually large+significant;
# the rest were not distinguishable from noise. Fewer parameters may generalize
# better given the existing CV/LB gap on the old baseline.
#
# Self-contained: does its own idempotent merge/setup, doesn't require 27-07_8
# or 27-07_9 to have been sourced first (but does need 27-07_1 foundation).

library(survival)

demo_cols <- c("Case", "segment", "agea", "Urbind")
stopifnot(all(demo_cols %in% names(train)))

for (col in c("segment", "agea", "Urbind", "agea_z", "is_luxury_segment")) {
  if (col %in% names(train_long)) train_long[, (col) := NULL]
}

demo_lookup <- unique(train[, ..demo_cols])
stopifnot(nrow(demo_lookup) == uniqueN(train_long$Case))
train_long <- merge(train_long, demo_lookup, by = "Case", sort = FALSE)

train_long[, agea_z := as.numeric(scale(agea))]

# ---- Verify segment levels before building the indicator (avoid guessing) ----
seg_levels <- unique(train_long$segment)
cat("=== segment levels observed ===\n")
print(seg_levels)

train_long[, is_luxury_segment := as.numeric(grepl("Luxury", segment, ignore.case = TRUE))]

cat("\n=== is_luxury_segment coverage ===\n")
print(train_long[, .(n_rows = .N, n_cases = uniqueN(Case)), by = is_luxury_segment])
cat("\nLevels flagged as luxury:\n")
print(unique(train_long[is_luxury_segment == 1, segment]))
cat("Levels NOT flagged as luxury:\n")
print(unique(train_long[is_luxury_segment == 0, segment]))

# ---- Model: baseline + Price:agea_z + Price:is_luxury_segment + Price:Urbind ----
rhs_vars <- c(identified_vars, "Price")
simple_formula <- as.formula(
  paste0("chosen ~ ", paste(rhs_vars, collapse = " + "),
         " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + strata(chid)")
)
simple_score_formula <- as.formula(
  paste("~", paste(rhs_vars, collapse = " + "),
        "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind")
)

fold_loss_simple <- numeric(5)
for (i in 1:5) {
  tr_long_i <- train_long[fold != i]
  va_long_i <- copy(train_long[fold == i])
  
  fit_simple <- tryCatch(
    clogit(simple_formula, data = tr_long_i, method = "exact"),
    error = function(e) {
      cat("Fold", i, "- clogit failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(fit_simple)) { fold_loss_simple[i] <- NA; next }
  
  if (any(is.na(coef(fit_simple)))) {
    cat("Fold", i, "- WARNING: aliased coefficient(s):",
        paste(names(coef(fit_simple))[is.na(coef(fit_simple))], collapse = ", "), "\n")
  }
  
  mm <- model.matrix(simple_score_formula, data = va_long_i)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  keep_coef <- names(coef(fit_simple))[!is.na(coef(fit_simple))]
  mm <- mm[, keep_coef, drop = FALSE]
  va_long_i[, score := as.numeric(mm %*% coef(fit_simple)[keep_coef])]
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
  
  fold_loss_simple[i] <- mlogloss(actual, pred_mat)
  cat("clogit+simplified_segment fold", i, "log-loss:", fold_loss_simple[i], "\n")
}

cat("\nclogit + Price:{agea_z, is_luxury_segment, Urbind} — CV summary\n")
cat("Mean:", mean(fold_loss_simple, na.rm = TRUE), " SD:", sd(fold_loss_simple, na.rm = TRUE),
    "(", sum(is.na(fold_loss_simple)), "fold(s) skipped)\n")
cat("Per-fold:", paste(round(fold_loss_simple, 5), collapse = ", "), "\n")
cat("vs full-segment interaction model (Registry #11): 1.164339\n")
cat("vs original baseline (no interactions):           1.180861\n")