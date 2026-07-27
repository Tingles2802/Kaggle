# 27-07_6_alt4_check.R
# Step 1 of the alt-4 investigation (see experiment_log Open Questions / Gotcha #4).
# Purpose: confirm or refute whether alt-4 is a fixed reference/status-quo/"no-choice"
# option in the conjoint design, before deciding whether it needs an ASC.
#
# Checks:
#   (a) What value are the 19 constant attr4 columns actually fixed at, in train?
#   (b) Is Price4 also constant in train? (If yes -> alt-4 is a fully fixed profile.)
#   (c) What is Ch4's overall choice share in train? (If people still pick a profile
#       that never changes, that's the signature of a reference/opt-out alternative.)
#   (d) Does test.csv show the same pattern (attr4 + Price4 constant)?
#
# Run 27-07_1_pipeline_foundation.R first (needs `train`, `attr_vars`).

# ---- (a)+(b): constant-value check on train, all 20 alt-4 columns (19 attrs + Price) ----
alt4_cols <- paste0(attr_vars, "4")   # attr_vars already includes "Price" per foundation script
alt4_summary <- sapply(alt4_cols, function(v) {
  vals <- unique(train[[v]])
  if (length(vals) == 1) paste0("CONSTANT = ", vals) else paste0(length(vals), " distinct values")
})
cat("=== Alt-4 columns in train (attrs + Price) ===\n")
print(alt4_summary)

# ---- (c): Ch4 choice share in train ----
ch_share <- colMeans(train[, .(Ch1, Ch2, Ch3, Ch4)])
cat("\n=== Choice share by alternative (train) ===\n")
print(ch_share)

# ---- (d): same check on test.csv ----
test_raw <- fread("test.csv")
alt4_summary_test <- sapply(alt4_cols, function(v) {
  vals <- unique(test_raw[[v]])
  if (length(vals) == 1) paste0("CONSTANT = ", vals) else paste0(length(vals), " distinct values")
})
cat("\n=== Alt-4 columns in test.csv (attrs + Price) ===\n")
print(alt4_summary_test)

# ---- Cross-check: same constant value in train vs test? ----
same_val_flag <- sapply(alt4_cols, function(v) {
  tr_vals <- unique(train[[v]]); te_vals <- unique(test_raw[[v]])
  if (length(tr_vals) == 1 && length(te_vals) == 1) {
    identical(as.character(tr_vals), as.character(te_vals))
  } else NA
})
cat("\n=== Train vs test: same constant value where both are constant? ===\n")
print(same_val_flag)

cat("\n=== Interpretation guide ===\n")
cat("If all attr4 + Price4 are CONSTANT in both train and test at the same value,\n")
cat("and Ch4's choice share is non-trivial (not ~0), alt-4 is very likely a fixed\n")
cat("reference/status-quo/no-choice option -> proceed to Step 2 (clogit + alt-4 ASC).\n")