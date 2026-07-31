# 31-07_6_segment_price_interaction.R (v2, self-contained)
# PREREQS (source first, same session): 27-07_1, 27-07_10
# score_clogit_cv() must also be defined - source 27-07_13 if missing (it will error
# on 'simple_formula' at its own line 54, that's expected/harmless - the function is
# defined before that line runs).
# Does NOT need 27-07_12 or 28-07_5 - avoids their unrelated dependencies (ranger_data/xgb).

library(survival); library(data.table)

if (!exists("score_clogit_cv")) stop("source 27-07_13 first (ok if it errors after defining score_clogit_cv)")
if (!exists("rhs_vars")) stop("source 27-07_10 first (need rhs_vars/agea_z/is_luxury_segment/Urbind)")

# build train_long_ext locally if not already present this session
if (!exists("train_long_ext")) {
  extra_demo <- unique(train[, .(Case, educ, region, gender)])
  train_long_ext <- merge(train_long, extra_demo, by = "Case", sort = FALSE)
  train_long_ext[, educ  := factor(educ,  levels = sort(unique(educ)))]
  train_long_ext[, region:= factor(region,levels = sort(unique(region)))]
  train_long_ext[, gender:= factor(gender,levels = sort(unique(gender)))]
}
if (!"segment" %in% names(train_long_ext)) {
  seg_lookup <- unique(train[, .(Case, segment)])
  train_long_ext <- merge(train_long_ext, seg_lookup, by = "Case", sort = FALSE)
}
train_long_ext[, segment := factor(segment)]

actual_lookup <- train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]

# Model 15 spec (copied from 28-07_5's confirmed Step 0)
model15_formula <- as.formula(paste0(
  "chosen ~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender + strata(chid)"
))
model15_score_formula <- as.formula(paste0(
  "~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender"
))

base_cv <- score_clogit_cv(model15_formula, model15_score_formula, train_long_ext, actual_lookup)
cat(sprintf("Model 15 reproduction: mean %.6f (expect 1.162469)\n", mean(base_cv)))
stopifnot(abs(mean(base_cv) - 1.162469) < 1e-4)  # STOP here if mismatch - don't trust below

formula_seg       <- update(model15_formula, . ~ . + Price:segment)
score_formula_seg <- update(model15_score_formula, ~ . + Price:segment)

cv_seg <- score_clogit_cv(formula_seg, score_formula_seg, train_long_ext, actual_lookup)
cat(sprintf("Model 15 + Price:segment: mean %.6f\n", mean(cv_seg)))