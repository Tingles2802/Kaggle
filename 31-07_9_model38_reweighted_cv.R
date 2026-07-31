# PASTE INTO: separate R session from the verification one (still counts as
# concurrent work, just not fully independent of the pipeline).
# PREREQS: run in a session where you've already sourced the chain needed for
# 31-07_5_segment_oof_slice.R (27-07_1, 27-07_4, ... through d_train_ext2/
# feat_cols_ext2/fold_vec), then source 31-07_5_segment_oof_slice.R itself.
# This script picks up right after that - needs oof_dt, seg_summary, train
# in memory.

library(data.table)

stopifnot(exists("oof_dt"), exists("seg_summary"), exists("train"))

# 1. Unweighted CV sanity check - should equal 1.148969 (already printed by
# 31-07_5, repeated here for convenience)
cat("Unweighted CV (sanity check):", mean(oof_dt$ll), "\n")

# 2. Train-side segment shares (from train, already in memory)
train_shares <- train[, .N, by = segment][, train_share := N / sum(N)][, .(segment, train_share)]

# 3. Test-side segment shares (fresh, targeted read - does NOT touch/overwrite
# the pipeline's `train` object, only reads test.csv for one column)
test_seg <- fread("test.csv", select = c("segment"))
test_shares <- test_seg[, .N, by = segment][, test_share := N / sum(N)][, .(segment, test_share)]

# 4. Combine with per-segment logloss from 31-07_5's seg_summary
seg_table <- merge(seg_summary, train_shares, by = "segment")
seg_table <- merge(seg_table, test_shares, by = "segment")
print(seg_table)

# 5. Reweighted CV: weight each segment's mean OOF logloss by test_share
reweighted_cv <- seg_table[, sum(mean_logloss * test_share) / sum(test_share)]
cat("\nTest-share-reweighted CV (Model 38):", reweighted_cv, "\n")
cat("Actual LB: 1.196   |   Raw CV: 1.148969\n")
cat("Reweighted CV vs LB gap:", abs(reweighted_cv - 1.196), "\n")
cat("(Compare to base-model reweighted deltas already logged: clogit15 +0.0444, ts +0.0378, xgb20 +0.0713, ranger18 +0.0805, glmnet30 +0.0846)\n")
