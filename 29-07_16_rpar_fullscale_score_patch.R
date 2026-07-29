# 29-07_16_rpar_fullscale_score_patch.R
# Run in the SAME console as 29-07_15 (reuses fit_full1, pred_test, ml_test,
# test_wide still in memory -- do not re-run the 88s fit).
# Fix: chid/Case/Task are plain columns on the dfidx object (per 27-07_12's
# header note #4), not accessed via index() -- that function doesn't apply
# to this mlogit version and was my error, not a package/data problem.

mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

df_test <- as.data.frame(ml_test)
stopifnot(all(c("chid","Case","Task") %in% names(df_test)))

chid_order <- unique(df_test$chid)  # first-appearance order
stopifnot(length(chid_order) == nrow(pred_test))

lookup <- unique(df_test[, c("chid","Case","Task")])
lookup <- lookup[match(chid_order, lookup$chid), ]
stopifnot(nrow(lookup) == nrow(pred_test), !anyNA(lookup$Case))

actual_lookup <- test_wide[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]
aligned <- merge(data.table::data.table(Case = lookup$Case, Task = lookup$Task),
                  actual_lookup, by = c("Case","Task"), sort = FALSE)
stopifnot(nrow(aligned) == nrow(pred_test))

actual_mat <- as.matrix(aligned[, .(Ch1, Ch2, Ch3, Ch4)])
fold1_loss <- mlogloss(actual_mat, pred_test)
cat("Fold 1 log-loss (rpar, R=50, full scale):", fold1_loss, "\n")
cat("vs clogit fold benchmark (Model 5, no interactions): ~1.18 mean\n")
