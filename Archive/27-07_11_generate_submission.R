# 27-07_11_generate_submission.R
# Fits Model 12 (current best, CV 1.162971: clogit baseline + Price:agea_z +
# Price:is_luxury_segment + Price:Urbind) on the FULL train set, scores test.csv,
# and writes submission_27-07_1.csv in sample_submission.csv format.
#
# Self-contained: only requires 27-07_1_pipeline_foundation.R sourced first
# (needs train, train_long, attr_vars, cat_attr_vars, identified_vars). Does NOT
# assume any of the interim scripts (27-07_8/9/10) were run — rebuilds all demo/
# interaction columns from scratch so there's no dependency on in-memory state.
#
# IMPORTANT: test's Price and agea must be scaled using TRAIN's center/scale
# (not re-scaled independently on test) — this script captures those params
# explicitly rather than assuming any existing scaled column is trustworthy.

library(survival)
library(data.table)

# =============================================================================
# STEP 1: rebuild train_long's demo + interaction columns, capture scaling params
# =============================================================================
demo_cols <- c("Case", "segment", "agea", "Urbind")
stopifnot(all(demo_cols %in% names(train)))

for (col in c("segment", "agea", "Urbind", "agea_z", "is_luxury_segment")) {
  if (col %in% names(train_long)) train_long[, (col) := NULL]
}
demo_lookup <- unique(train[, ..demo_cols])
stopifnot(nrow(demo_lookup) == uniqueN(train_long$Case))
train_long <- merge(train_long, demo_lookup, by = "Case", sort = FALSE)

agea_scaled  <- scale(train_long$agea)
train_long[, agea_z := as.numeric(agea_scaled)]
agea_center  <- attr(agea_scaled, "scaled:center")
agea_scale_v <- attr(agea_scaled, "scaled:scale")

train_long[, is_luxury_segment := as.numeric(grepl("Luxury", segment, ignore.case = TRUE))]

# Price's scaling params: recompute independently from raw wide Price1-4 columns
# (train_long$Price itself was already scaled by pipeline_foundation.R — using it
# directly for the fit is fine and matches all prior CV runs, but we need the
# center/scale numbers themselves to transform test's Price the same way).
price_raw_train <- unlist(lapply(1:4, function(a) train[[paste0("Price", a)]]))
price_center <- mean(price_raw_train)
price_scale_v <- sd(price_raw_train)

# =============================================================================
# STEP 2: fit Model 12 on the FULL train set (no CV split — final model)
# =============================================================================
rhs_vars <- c(identified_vars, "Price")
final_formula <- as.formula(
  paste0("chosen ~ ", paste(rhs_vars, collapse = " + "),
         " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + strata(chid)")
)
final_score_formula <- as.formula(
  paste("~", paste(rhs_vars, collapse = " + "),
        "+ Price:agea_z + Price:is_luxury_segment + Price:Urbind")
)

fit_final <- clogit(final_formula, data = train_long, method = "exact")

if (any(is.na(coef(fit_final)))) {
  stop("Aliased coefficient(s) in final fit — investigate before generating a submission: ",
       paste(names(coef(fit_final))[is.na(coef(fit_final))], collapse = ", "))
}
cat("Final model fit on full train (", nrow(train), "rows ) — coefficients:\n")
print(round(coef(fit_final), 5))

# =============================================================================
# STEP 3: build test_long, applying TRAIN's scaling params (not test's own)
# =============================================================================
test_raw <- fread("test.csv")

build_long_test <- function(dt, alt) {
  cols <- paste0(attr_vars, alt)
  out <- dt[, ..cols]
  setnames(out, cols, attr_vars)
  out[, `:=`(
    Case = dt$Case,
    Task = dt$Task,
    No   = dt$No,
    chid = paste0(dt$Case, "_", dt$Task),
    alt  = alt
  )]
  out
}
test_long <- rbindlist(lapply(1:4, build_long_test, dt = test_raw))
setorder(test_long, chid, alt)

# Factor levels must match train_long exactly (guarantees identical model.matrix
# columns even if a level happens not to appear in test's data).
for (v in cat_attr_vars) {
  test_long[, (v) := factor(get(v), levels = levels(train_long[[v]]))]
}
unseen_check <- sapply(cat_attr_vars, function(v) sum(is.na(test_long[[v]])))
if (any(unseen_check > 0)) {
  stop("Unseen category level(s) in test.csv not present in train — investigate: ",
       paste(names(unseen_check)[unseen_check > 0], collapse = ", "))
}

test_long[, Price := as.numeric(Price)]
test_long[, Price := (Price - price_center) / price_scale_v]

demo_cols_test <- c("Case", "segment", "agea", "Urbind")
stopifnot(all(demo_cols_test %in% names(test_raw)))
demo_lookup_test <- unique(test_raw[, ..demo_cols_test])
stopifnot(nrow(demo_lookup_test) == uniqueN(test_long$Case))
test_long <- merge(test_long, demo_lookup_test, by = "Case", sort = FALSE)

test_long[, agea_z := (agea - agea_center) / agea_scale_v]
test_long[, is_luxury_segment := as.numeric(grepl("Luxury", segment, ignore.case = TRUE))]

# =============================================================================
# STEP 4: score test_long from fitted coefficients (predict.coxph doesn't handle
# brand-new strata/chid — same score-by-hand approach used throughout Day 3 CV)
# =============================================================================
mm <- model.matrix(final_score_formula, data = test_long)
mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
keep_coef <- names(coef(fit_final))
mm <- mm[, keep_coef, drop = FALSE]
test_long[, score := as.numeric(mm %*% coef(fit_final)[keep_coef])]
test_long[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]

setorder(test_long, chid, alt)
preds_wide <- dcast(test_long, chid ~ alt, value.var = "prob")
setnames(preds_wide, as.character(1:4), c("Ch1","Ch2","Ch3","Ch4"))

no_lookup <- unique(test_long[, .(chid, No)])
preds_wide <- merge(no_lookup, preds_wide, by = "chid", sort = FALSE)

cat("\nRow-sum check (should all be ~1):\n")
print(summary(rowSums(preds_wide[, .(Ch1, Ch2, Ch3, Ch4)])))
cat("\nAny NA/NaN probabilities?", any(is.na(preds_wide[, .(Ch1, Ch2, Ch3, Ch4)])), "\n")

# =============================================================================
# STEP 5: order to match sample_submission.csv exactly, write file
# =============================================================================
sample_sub <- fread("sample_submission.csv")
stopifnot(nrow(sample_sub) == nrow(preds_wide))
submission <- merge(sample_sub[, .(No)], preds_wide[, .(No, Ch1, Ch2, Ch3, Ch4)],
                    by = "No", sort = FALSE)
stopifnot(nrow(submission) == nrow(sample_sub))
stopifnot(all(submission$No == sample_sub$No))

fwrite(submission, "submission_27-07_1.csv")
cat("\nWrote submission_27-07_1.csv —", nrow(submission), "rows.\n")
print(head(submission))