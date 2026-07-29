# 29-07_15_rpar_fullscale_singlefold.R
# STANDALONE — no other R files need to be run first.
# DIAGNOSTIC ONLY (Section 1.2): fold 1 only, at full data scale (~17k train
# rows), to check convergence + runtime before committing to a full 5-fold
# CV script. gmnl crashed only at full scale (Gotcha #24) despite passing a
# small-slice diagnostic -- checking rpar isn't the same before investing in
# 5 folds.

library(data.table)
library(mlogit)

set.seed(2024)

# ---- 1. Load + identify vars ----
dt <- fread("train.csv")

cat_attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP",
               "KA","SC","TS","NV","MA","LB","AF","HU")
identified_vars <- c(cat_attrs, "Price")

dt[, choice_col := fifelse(Ch1 == 1, "1",
                           fifelse(Ch2 == 1, "2",
                                   fifelse(Ch3 == 1, "3", "4")))]

varying_cols <- unlist(lapply(identified_vars, function(v) paste0(v, 1:4)))
stopifnot(all(varying_cols %in% names(dt)))

# ---- 2. Canonical Case-grouped 5-fold split (seed 2024, matches 27-07_1) ----
unique_cases <- unique(dt$Case)
case_folds <- data.table(
  Case = unique_cases,
  fold = sample(rep(1:5, length.out = length(unique_cases)))
)
dt <- merge(dt, case_folds, by = "Case", sort = FALSE)

# ---- 3. Fold 1 only, full scale ----
train_wide <- dt[fold != 1]
test_wide  <- dt[fold == 1]

cat("train_wide:", nrow(train_wide), "rows /", uniqueN(train_wide$Case), "cases\n")
cat("test_wide: ", nrow(test_wide), "rows /", uniqueN(test_wide$Case), "cases\n")

ml_train <- mlogit.data(
  as.data.frame(train_wide), shape = "wide", choice = "choice_col",
  varying = which(names(train_wide) %in% varying_cols), sep = "", id.var = "Case"
)
ml_test <- mlogit.data(
  as.data.frame(test_wide), shape = "wide", choice = "choice_col",
  varying = which(names(test_wide) %in% varying_cols), sep = "", id.var = "Case"
)

f <- mFormula(as.formula(paste("choice_col ~", paste(identified_vars, collapse = " + "), "| 0")))

cat("\n=== Starting full-scale fit (fold 1, R=50) — timing below ===\n")
fit_time <- system.time({
  fit_full1 <- tryCatch(
    mlogit(f, data = ml_train, rpar = c(Price = "n"), R = 50,
           halton = NA, panel = TRUE, print.level = 1),
    error = function(e) e
  )
})

if (inherits(fit_full1, "error")) {
  cat("FULL-SCALE FIT FAILED:", conditionMessage(fit_full1), "\n")
} else {
  cat("\nFit succeeded. Elapsed time:\n")
  print(fit_time)
  print(summary(fit_full1))

  cat("\n=== predict() on held-out fold ===\n")
  pred_time <- system.time({
    pred_test <- tryCatch(predict(fit_full1, newdata = ml_test), error = function(e) e)
  })
  if (inherits(pred_test, "error")) {
    cat("predict() FAILED:", conditionMessage(pred_test), "\n")
  } else {
    cat("predict() succeeded. dim:", paste(dim(pred_test), collapse = "x"), "\n")
    print(pred_time)
    cat("Any NA:", anyNA(pred_test), " | Row sums range:",
        paste(round(range(rowSums(pred_test)), 6), collapse = " to "), "\n")

    # ---- 4. Score fold 1 log-loss (need Case+Task alignment, not raw predict() row order) ----
    mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
      pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
      -mean(rowSums(actual_mat * log(pred_mat)))
    }
    # predict() returns one row per chid in ml_test's internal order -- recover
    # Case/Task alignment via the index stored on ml_test rather than assuming
    # row order matches test_wide (do not assume; verify).
    idx <- index(ml_test)
    chid_order <- unique(idx$chid)
    lookup <- unique(as.data.frame(ml_test)[, c("chid", "Case", "Task")])
    lookup <- lookup[match(chid_order, lookup$chid), ]
    stopifnot(nrow(lookup) == nrow(pred_test))

    actual_lookup <- test_wide[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]
    aligned <- merge(data.table(Case = lookup$Case, Task = lookup$Task),
                      actual_lookup, by = c("Case","Task"), sort = FALSE)
    stopifnot(nrow(aligned) == nrow(pred_test))

    actual_mat <- as.matrix(aligned[, .(Ch1, Ch2, Ch3, Ch4)])
    fold1_loss <- mlogloss(actual_mat, pred_test)
    cat("\nFold 1 log-loss (rpar, R=50):", fold1_loss, "\n")
    cat("vs clogit fold benchmark range (Model 5, no interactions): ~1.18 mean\n")
  }
}
