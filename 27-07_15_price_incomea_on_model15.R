# REQUIRES: 27-07_1, 27-07_10, 27-07_13, 27-07_14 already sourced, in that order
# (14 defines base_terms, used below - verified by inspecting 14's source).
# Needs train_long_ext, score_clogit_cv(), rhs_vars, actual_lookup, base_terms in scope.

# --- Step 0: verification (Section 1 pipeline-refactor rule) ---
# Must reproduce Model 15 = 1.162469 (sd 0.015443) before trusting anything below.
verify_f       <- as.formula(paste0("chosen ~ ", paste(rhs_vars, collapse=" + "), base_terms, " + Price:gender + strata(chid)"))
verify_score_f <- as.formula(paste0("~ ", paste(rhs_vars, collapse=" + "), base_terms, " + Price:gender"))
verify_scores  <- score_clogit_cv(verify_f, verify_score_f, train_long_ext, actual_lookup)
cat(sprintf("\nVerification mean: %.6f | sd: %.6f (expect 1.162469 / 0.015443)\n\n",
            mean(verify_scores), sd(verify_scores)))

# --- Step 1: merge incomea in (not present in train_long_ext), standardize like agea_z ---
extra_income <- unique(train[, .(Case, incomea)])
train_long_ext2 <- merge(train_long_ext, extra_income, by = "Case", sort = FALSE)
train_long_ext2[, incomea_z := as.numeric(scale(incomea))]

# --- Step 2: test Price:incomea_z on top of Model 15 (new combo, not previously tested -
# Price:incomea was only tested alone on the base model in Model 6, ruled out there) ---
model16_formula <- as.formula(paste0(
  "chosen ~ ", paste(rhs_vars, collapse = " + "), base_terms,
  " + Price:gender + Price:incomea_z + strata(chid)"
))
model16_score_formula <- as.formula(paste0(
  "~ ", paste(rhs_vars, collapse = " + "), base_terms,
  " + Price:gender + Price:incomea_z"
))

model16_scores <- score_clogit_cv(model16_formula, model16_score_formula, train_long_ext2, actual_lookup)
cat(sprintf("\nModel 16 (Model 15 + Price:incomea_z) mean: %.6f | sd: %.6f\n",
            mean(model16_scores), sd(model16_scores)))

# Adopt only if it beats Model 15 (1.162469 / 0.015443) with comparable/lower sd -
# not just a lower mean with higher variance (same bar as educ/region rejection).
