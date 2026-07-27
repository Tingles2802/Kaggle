# REQUIRES (verified minimal chain, Gotcha #14): rm(list=ls()), then source
# 27-07_1, 27-07_10, 27-07_13 in that order. Needs train_long_ext, score_clogit_cv(),
# rhs_vars, actual_lookup in scope from 27-07_13 — confirm actual_lookup exists
# before running (name unverified, see Gotcha #12).# REQUIRES: same environment as 27-07_13 (already verified/extended there).
# Needs train_long_ext (with fixed-level educ/region/gender factors) and
# score_clogit_cv(), rhs_vars, actual_lookup already in scope from 27-07_13.
# If not in scope, re-run 27-07_13 first (no need to rm/re-source beyond that).

base_terms <- "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind"

make_formula <- function(extra_term, score_side = FALSE) {
  lhs <- if (score_side) "~ " else "chosen ~ "
  tail <- if (score_side) "" else " + strata(chid)"
  as.formula(paste0(lhs, paste(rhs_vars, collapse = " + "), base_terms, extra_term, tail))
}

terms_to_test <- list(
  educ_only   = " + Price:educ",
  region_only = " + Price:region",
  gender_only = " + Price:gender"
)

for (nm in names(terms_to_test)) {
  fit_f   <- make_formula(terms_to_test[[nm]], score_side = FALSE)
  score_f <- make_formula(terms_to_test[[nm]], score_side = TRUE)
  scores  <- score_clogit_cv(fit_f, score_f, train_long_ext, actual_lookup)
  cat(sprintf("\n%s -> mean: %.6f | sd: %.6f\n", nm, mean(scores), sd(scores)))
}

# Compare each to Model 12 (1.162971 / 0.01546). Keep only a term that beats it
# with comparable/lower variance - not just a lower mean with much higher sd.
