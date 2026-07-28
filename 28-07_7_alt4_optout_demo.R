# Requires: 27-07_10 then 27-07_13 already sourced this session (train_long_ext,
# score_clogit_cv(), actual_lookup, rhs_vars, is_luxury_segment, agea_z, gender factor).
# Purpose: Model 10's plain asc4 aliased (confirmed). Test alt-4 opt-out utility as
#          demographic interactions only (no bare intercept).
# v2 fix: sanity formula was missing Model 12's agea_z/is_luxury_segment/Urbind terms
#         (Model 15 = Model 12 spec + Price:gender, not rhs_vars + Price:gender alone).
#         Also: is_alt4:gender aliased because factor interaction w/o main effect gets
#         full-rank dummy coding, recreating the plain-ASC collinearity. Fixed via
#         numeric is_male dummy instead of the gender factor.

# --- Model 15 base spec (Model 12 + Price:gender) ---
m15_formula <- as.formula(paste(
  "chosen ~", paste(rhs_vars, collapse = " + "),
  "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender + strata(chid)"
))
m15_score_formula <- as.formula(paste(
  "~", paste(rhs_vars, collapse = " + "),
  "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender"
))
m15_scores <- score_clogit_cv(m15_formula, m15_score_formula, train_long_ext, actual_lookup)
cat(sprintf("Model 15 reproduction: %.6f | sd %.6f (expect ~1.162469)\n",
            mean(m15_scores), sd(m15_scores)))
stopifnot(abs(mean(m15_scores) - 1.162469) < 0.005)

# --- alt-4 opt-out indicator + numeric gender dummy (avoids full-rank factor aliasing) ---
train_long_ext[, is_alt4 := as.integer(alt == 4)]
train_long_ext[, is_male := as.integer(gender == "Male")]

alt4_terms <- c(
  "is_alt4:is_male",
  "is_alt4:is_luxury_segment",
  "is_alt4:agea_z"
)

alt4_formula <- as.formula(paste(
  "chosen ~", paste(rhs_vars, collapse = " + "),
  "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
  "+", paste(alt4_terms, collapse = " + "),
  "+ strata(chid)"
))
alt4_score_formula <- as.formula(paste(
  "~", paste(rhs_vars, collapse = " + "),
  "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
  "+", paste(alt4_terms, collapse = " + ")
))

alt4_scores <- score_clogit_cv(alt4_formula, alt4_score_formula, train_long_ext, actual_lookup)
cat(sprintf("Alt-4 opt-out (demo interactions) CV: %.6f | sd %.6f\n",
            mean(alt4_scores), sd(alt4_scores)))
cat("Per-fold:", paste(round(alt4_scores, 5), collapse = ", "), "\n")

# --- check for aliasing on full data ---
fit_full <- clogit(alt4_formula, data = train_long_ext, method = "exact")
print(summary(fit_full)$coefficients)
# Check the is_alt4:* rows above specifically for NA coef / se=0