# Requires: 27-07_1, 27-07_10, 27-07_13 already sourced this session (rhs_vars,
# identified_vars, train_long_ext, actual_lookup, score_clogit_cv() must exist).
# One-shot screen: does gender x attribute-level heterogeneity help, on top of Model 15?

stopifnot(exists("rhs_vars"), exists("identified_vars"), exists("train_long_ext"),
          exists("actual_lookup"), exists("score_clogit_cv"))

# Model 15 base spec + gender:each attribute (group term, not isolated per-attribute yet)
attr_gender_terms <- paste0("gender:", identified_vars, collapse = " + ")

fit_formula_24 <- as.formula(paste(
  "chosen ~", paste(rhs_vars, collapse = " + "),
  "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
  "+", attr_gender_terms,
  "+ strata(chid)"
))

score_formula_24 <- as.formula(paste(
  "~", paste(rhs_vars, collapse = " + "),
  "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
  "+", attr_gender_terms
))

cv_24 <- score_clogit_cv(fit_formula_24, score_formula_24, train_long_ext, actual_lookup)
print(cv_24)  # compare against Model 15: 1.162469

# If clogit throws convergence warnings/errors (19 extra interaction terms is a lot),
# paste the exact error back before I suggest reducing the term set.