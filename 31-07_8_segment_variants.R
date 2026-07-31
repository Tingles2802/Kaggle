# 31-07-8_segment_variants.R
# Run AFTER 31-07_6_segment_price_interaction.R has completed successfully
# Tests Price:segment variants WITHOUT is_luxury_segment aliasing

library(data.table)
library(survival)

# Check that objects from _6 exist
required_objects <- c("train", "train_long_ext", "rhs_vars", 
                      "model15_formula", "model15_score_formula",
                      "score_clogit_cv", "actual_lookup", "base_cv")
missing <- required_objects[!sapply(required_objects, exists)]
if (length(missing) > 0) {
  cat("ERROR: Missing objects:", paste(missing, collapse=", "), "\n")
  cat("Run 31-07_6_segment_price_interaction.R first\n")
  stop("Missing prerequisites")
}

# Ensure Price_z exists in train_long_ext
if (!"Price_z" %in% names(train_long_ext)) {
  cat("Creating Price_z in train_long_ext...\n")
  train_long_ext[, Price_z := as.numeric(scale(Price))]
}

# Ensure segment is in train_long_ext
if (!"segment" %in% names(train_long_ext)) {
  cat("Adding segment to train_long_ext...\n")
  seg_lookup <- unique(train[, .(Case, segment)])
  train_long_ext <- merge(train_long_ext, seg_lookup, by = "Case", sort = FALSE)
  train_long_ext[, segment := factor(segment)]
}

# Ensure is_luxury_segment is in train_long_ext
if (!"is_luxury_segment" %in% names(train_long_ext)) {
  cat("Adding is_luxury_segment to train_long_ext...\n")
  luxury_segments <- c("Prestige Luxury Sedan", "Midsize Luxury Utility segements")
  train_long_ext[, is_luxury_segment := as.integer(segment %in% luxury_segments)]
}

cat("\n========================================\n")
cat("Testing Price:segment variants (no luxury aliasing)\n")
cat("========================================\n\n")

cat("Baseline Model 15 CV (from _6):\n")
cat(sprintf("  mean: %.6f\n", mean(base_cv)))
cat("  folds:", paste(round(base_cv, 6), collapse=", "), "\n\n")

# ------------------------------------------------------------------------------
# Variant A: Price:segment replaces is_luxury_segment (no aliasing)
# ------------------------------------------------------------------------------

cat("Variant A: Price:segment REPLACES is_luxury_segment\n")
cat("  (removes Price_z:is_luxury_segment, adds Price_z:segment)\n")

# Build formulas explicitly (avoid update() issues)
rhs_vars_with_price <- c(rhs_vars, "Price_z")

# Variant A: Replace is_luxury_segment with segment
varA_rhs <- paste(
  rhs_vars_with_price,
  "+ Price_z:gender + Price_z:agea_z + Price_z:segment + Price_z:Urbind",
  collapse = " + "
)
varA_formula <- as.formula(paste("chosen ~", varA_rhs, "+ strata(chid)"))
varA_score <- as.formula(paste("~", varA_rhs))

varA_cv <- score_clogit_cv(varA_formula, varA_score, train_long_ext, actual_lookup)
cat(sprintf("  CV mean: %.6f\n", mean(varA_cv, na.rm = TRUE)))
cat("  folds:", paste(round(varA_cv, 6), collapse=", "), "\n")
cat(sprintf("  Delta vs baseline: %.6f\n\n", mean(varA_cv, na.rm = TRUE) - mean(base_cv)))

# ------------------------------------------------------------------------------
# Variant B: segment main effect + Price:segment
# ------------------------------------------------------------------------------

cat("Variant B: segment main effect + Price:segment\n")
cat("  (adds segment main effects to Variant A)\n")

varB_rhs <- paste(
  rhs_vars_with_price,
  "+ segment + Price_z:gender + Price_z:agea_z + Price_z:segment + Price_z:Urbind",
  collapse = " + "
)
varB_formula <- as.formula(paste("chosen ~", varB_rhs, "+ strata(chid)"))
varB_score <- as.formula(paste("~", varB_rhs))

varB_cv <- score_clogit_cv(varB_formula, varB_score, train_long_ext, actual_lookup)
cat(sprintf("  CV mean: %.6f\n", mean(varB_cv, na.rm = TRUE)))
cat("  folds:", paste(round(varB_cv, 6), collapse=", "), "\n")
cat(sprintf("  Delta vs baseline: %.6f\n\n", mean(varB_cv, na.rm = TRUE) - mean(base_cv)))

# ------------------------------------------------------------------------------
# Variant C: BOTH is_luxury_segment AND Price:segment
# ------------------------------------------------------------------------------

cat("Variant C: BOTH is_luxury_segment AND Price:segment\n")
cat("  (adds Price_z:segment to baseline Model 15)\n")

varC_rhs <- paste(
  rhs_vars_with_price,
  "+ Price_z:gender + Price_z:agea_z + Price_z:is_luxury_segment + Price_z:segment + Price_z:Urbind",
  collapse = " + "
)
varC_formula <- as.formula(paste("chosen ~", varC_rhs, "+ strata(chid)"))
varC_score <- as.formula(paste("~", varC_rhs))

varC_cv <- score_clogit_cv(varC_formula, varC_score, train_long_ext, actual_lookup)
cat(sprintf("  CV mean: %.6f\n", mean(varC_cv, na.rm = TRUE)))
cat("  folds:", paste(round(varC_cv, 6), collapse=", "), "\n")
cat(sprintf("  Delta vs baseline: %.6f\n\n", mean(varC_cv, na.rm = TRUE) - mean(base_cv)))

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

cat("========================================\n")
cat("SUMMARY\n")
cat("========================================\n")
cat(sprintf("Baseline (Model 15):                 %.6f\n", mean(base_cv)))
cat(sprintf("Variant A (replace luxury w/segment): %.6f  (delta: %+.6f)\n", 
            mean(varA_cv, na.rm = TRUE), mean(varA_cv, na.rm = TRUE) - mean(base_cv)))
cat(sprintf("Variant B (segment + Price:segment):  %.6f  (delta: %+.6f)\n", 
            mean(varB_cv, na.rm = TRUE), mean(varB_cv, na.rm = TRUE) - mean(base_cv)))
cat(sprintf("Variant C (both luxury + segment):    %.6f  (delta: %+.6f)\n", 
            mean(varC_cv, na.rm = TRUE), mean(varC_cv, na.rm = TRUE) - mean(base_cv)))

# Determine best
variants <- data.table(
  name = c("baseline", "varA", "varB", "varC"),
  cv = c(mean(base_cv), mean(varA_cv, na.rm = TRUE), 
         mean(varB_cv, na.rm = TRUE), mean(varC_cv, na.rm = TRUE))
)
best <- variants[which.min(cv)]
cat(sprintf("\nBest: %s (%.6f)\n", best$name, best$cv))
if (best$name == "baseline") {
  cat("No variant beats the baseline Model 15.\n")
} else {
  cat(sprintf("Improvement over baseline: %+.6f\n", best$cv - mean(base_cv)))
}

# Save results
results <- data.table(
  variant = c("baseline", "varA", "varB", "varC"),
  cv_mean = c(mean(base_cv), mean(varA_cv, na.rm = TRUE), 
              mean(varB_cv, na.rm = TRUE), mean(varC_cv, na.rm = TRUE)),
  cv_sd = c(sd(base_cv), sd(varA_cv, na.rm = TRUE), 
            sd(varB_cv, na.rm = TRUE), sd(varC_cv, na.rm = TRUE)),
  fold1 = c(base_cv[1], varA_cv[1], varB_cv[1], varC_cv[1]),
  fold2 = c(base_cv[2], varA_cv[2], varB_cv[2], varC_cv[2]),
  fold3 = c(base_cv[3], varA_cv[3], varB_cv[3], varC_cv[3]),
  fold4 = c(base_cv[4], varA_cv[4], varB_cv[4], varC_cv[4]),
  fold5 = c(base_cv[5], varA_cv[5], varB_cv[5], varC_cv[5])
)
results[, delta := cv_mean - results[variant == "baseline", cv_mean]]

fwrite(results, "segment_variants_results.csv")
cat("\nResults saved to segment_variants_results.csv\n")
cat("\n=== Done ===\n")