# STEP 0 - DO THIS BEFORE ANYTHING ELSE (file explorer or console):
#   file.copy("submission_30-07_2.csv", "submission_30-07_2_ORIGINAL_BACKUP.csv")
# 30-07_10 fwrite()s to "submission_30-07_2.csv" - same name as your existing
# submitted file. Without this backup, re-running 30-07_10 overwrites the file
# you need to diff against.

# PASTE INTO: fresh RStudio console (NEW SESSION - do not reuse existing one)
# PREREQS: run in a session with NOTHING loaded. Then source, in order, the full
# Model 38 canonical chain (per experiment_log Sec 2.3 "Model 38 full chain"):
#   27-07_1 -> 4 -> 5 -> 10 -> 13 -> 28-07_5 -> 7 -> 11 -> 28-07_1 -> 12 ->
#   29-07_3 -> 4 -> 2 -> 10 -> 20 -> 30-07_5 -> 30-07_1 -> 29-07_19 (standalone,
#   run in its own slot per Gotcha #28) -> 30-07_3 -> 30-07_7 -> 30-07_8 -> 30-07_10
# 30-07_10 will re-overwrite submission_30-07_2.csv with a fresh refit -
# that's expected, it's what we're checking. Then run the code below.

# 1. Confirm CV reproduces exactly (results[1] = best grid row from 30-07_8)
cv_i <- results[1]$cv_loss
cat("Model 38 CV (fresh grid search):", cv_i, "\n")
cat("Expected: 1.148969\n")
cat("Match:", isTRUE(all.equal(cv_i, 1.148969, tolerance = 1e-5)), "\n")

# 2. Confirm fresh full-refit test predictions match the ORIGINAL submitted file
old_sub <- fread("Archive/submission_30-07_2.csv")  # adjust path to your archive folder
new_sub <- fread("submission_30-07_2.csv")  # freshly overwritten by 30-07_10 just now

setkey(old_sub, No)
setkey(new_sub, No)

diffs <- merge(old_sub, new_sub, by = "No", suffixes = c("_old", "_new"))
max_diff <- max(abs(diffs$Ch1_old - diffs$Ch1_new),
                abs(diffs$Ch2_old - diffs$Ch2_new),
                abs(diffs$Ch3_old - diffs$Ch3_new),
                abs(diffs$Ch4_old - diffs$Ch4_new))

cat("Max abs diff between old and fresh predictions:", max_diff, "\n")
cat("Clean match (should be ~0, allow tiny float noise):", max_diff < 1e-8, "\n")

# 3. Sanity: row count and sum-to-1 check
cat("Rows:", nrow(new_sub), "(expect 4997)\n")
cat("All rows sum to 1:", all(abs(rowSums(new_sub[, .(Ch1, Ch2, Ch3, Ch4)]) - 1) < 1e-6), "\n")
