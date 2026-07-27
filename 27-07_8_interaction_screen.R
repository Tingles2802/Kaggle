# 27-07_8_interaction_screen.R
# Screens 5 candidate demographic x Price interactions in ONE full-data clogit fit
# (no CV loop) — cheap way to see which, if any, look real before spending a full
# 5-fold CV round-trip on each. Only demographic x Price pairings tested (Price's
# meaning is unambiguous; attribute codes CC/GN/... are anonymized, so guessing at
# attribute-level interactions would just be speculation — see decision log).
# `Price:incomea` already ruled out (Model Registry #6) — not repeated here.
#
# Candidates: segment, agea (age), Urbind (urban/rural), milesa (annual mileage),
# nighta (night-driving frequency), each interacted with Price.
#
# Run 27-07_1_pipeline_foundation.R first, then this. Idempotent — safe to re-source.

library(survival)

demo_cols <- c("Case", "segment", "agea", "Urbind", "milesa", "nighta")
stopifnot(all(demo_cols %in% names(train)))

# Idempotency guard: drop previously-merged demo/interaction cols if present
for (col in c("segment", "agea", "Urbind", "milesa", "nighta",
              "agea_z", "milesa_z", "nighta_z")) {
  if (col %in% names(train_long)) train_long[, (col) := NULL]
}

demo_lookup <- unique(train[, ..demo_cols])
stopifnot(nrow(demo_lookup) == uniqueN(train_long$Case))

train_long <- merge(train_long, demo_lookup, by = "Case", sort = FALSE)

# Standardize continuous demo vars (interpretability only — clogit fit/CV is
# mathematically invariant to linear rescaling of a covariate, per the
# incomea lesson; Urbind left as-is, already a 0/1 indicator; segment is
# categorical, left as factor).
train_long[, agea_z   := as.numeric(scale(agea))]
train_long[, milesa_z := as.numeric(scale(milesa))]
train_long[, nighta_z := as.numeric(scale(nighta))]
train_long[, segment  := as.factor(segment)]

screen_formula <- as.formula(
  paste0("chosen ~ ", paste(c(identified_vars, "Price"), collapse = " + "),
         " + Price:segment + Price:agea_z + Price:Urbind + Price:milesa_z + Price:nighta_z",
         " + strata(chid)")
)

fit_screen <- clogit(screen_formula, data = train_long, method = "exact")

if (any(is.na(coef(fit_screen)))) {
  cat("WARNING: aliased coefficient(s):",
      paste(names(coef(fit_screen))[is.na(coef(fit_screen))], collapse = ", "), "\n\n")
}

coefs <- summary(fit_screen)$coefficients
interact_rows <- grepl("^Price:", rownames(coefs))
interact_coefs <- coefs[interact_rows, , drop = FALSE]
interact_coefs <- interact_coefs[order(interact_coefs[, "Pr(>|z|)"]), , drop = FALSE]

cat("=== Candidate Price interactions, sorted by p-value ===\n")
print(round(interact_coefs, 5))

cat("\n=== Screening guide ===\n")
cat("Look for: |z| notably > ~2 (p < 0.05) AND a coefficient magnitude large enough\n")
cat("to plausibly matter (compare to Price's own coefficient below), not just\n")
cat("statistical significance on 21,565 rows (large n makes tiny effects 'significant').\n")
cat("Price main effect coef:", coefs["Price", "coef"], "\n")
cat("\nOnly promising term(s) should get a full 5-fold CV run next — don't CV all 5.\n")