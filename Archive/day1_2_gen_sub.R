# =============================================================================
# Generate submission from Model B' (clogit), fit on FULL training data
# Run after day2_models_multinom_mlogit.R and day2_model_bp_clogit.R in the
# same session (reuses train_long, identified_vars, clogit_formula, attr_vars,
# cat_attr_vars, train).
# =============================================================================

library(survival)
library(data.table)

# ---- 1. Refit on ALL training data (no fold holdout) ---------------------
final_fit <- clogit(clogit_formula, data = train_long, method = "exact")
cat("Final model coefficients:\n")
print(coef(final_fit))

# ---- 2. Recover Price's training center/scale ----------------------------
# train_long$Price was overwritten in place by scale() in
# day2_models_multinom_mlogit.R (mean/sd not saved). Recompute them from the
# original wide Price1..Price4 columns in `train` -- same underlying values,
# just before they were stacked into long format, so mean/sd match exactly.
raw_train_price <- unlist(train[, .(Price1, Price2, Price3, Price4)])
price_center <- mean(raw_train_price)
price_scale  <- sd(raw_train_price)

# ---- 3. Load and reshape test data the same way as train -----------------
test <- fread("test.csv")

build_long_test <- function(dt, alt) {
  cols <- paste0(attr_vars, alt)
  out <- dt[, ..cols]
  setnames(out, cols, attr_vars)
  out[, `:=`(
    No   = dt$No,
    Case = dt$Case,
    Task = dt$Task,
    chid = paste0(dt$Case, "_", dt$Task),
    alt  = alt
  )]
  out
}

test_long <- rbindlist(lapply(1:4, build_long_test, dt = test))
setorder(test_long, chid, alt)

# Match factor levels exactly to train_long; standardize Price with TRAIN's
# center/scale (using test's own scale() would silently shift every score).
for (v in cat_attr_vars) {
  test_long[, (v) := factor(get(v), levels = levels(train_long[[v]]))]
}
test_long[, Price := as.numeric(Price)]
test_long[, Price := (Price - price_center) / price_scale]

# Sanity check: flag any row where a categorical level didn't match a
# known training level (would produce NA and break the score below).
na_check <- sapply(cat_attr_vars, function(v) sum(is.na(test_long[[v]])))
if (any(na_check > 0)) {
  cat("WARNING: unseen factor levels in test data for:\n")
  print(na_check[na_check > 0])
  stop("Resolve unseen levels before generating predictions.")
}

# ---- 4. Score test data and convert to probabilities ----------------------
score_formula <- as.formula(paste("~", paste(identified_vars, collapse = " + ")))
mm <- model.matrix(score_formula, data = test_long)
mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
mm <- mm[, names(coef(final_fit)), drop = FALSE]
test_long[, score := as.numeric(mm %*% coef(final_fit))]
test_long[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]

# ---- 5. Reshape to wide submission format ---------------------------------
setorder(test_long, chid, alt)
preds_wide <- dcast(test_long, chid ~ alt, value.var = "prob")
setnames(preds_wide, as.character(1:4), c("Ch1", "Ch2", "Ch3", "Ch4"))

no_lookup <- unique(test_long[, .(chid, No)])
preds_wide <- merge(no_lookup, preds_wide, by = "chid", sort = FALSE)

submission <- preds_wide[, .(No, Ch1, Ch2, Ch3, Ch4)]
setorder(submission, No)

# Sanity checks before writing
stopifnot(nrow(submission) == 4997)
row_sums <- rowSums(submission[, .(Ch1, Ch2, Ch3, Ch4)])
cat("Row sum range (should all be ~1):", range(row_sums), "\n")

fwrite(submission, "submission_26-07_2.csv")
cat("Wrote submission_26-07_2.csv --", nrow(submission), "rows\n")