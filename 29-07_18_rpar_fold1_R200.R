# 29-07_18_rpar_fold1_R200.R
# STANDALONE — no other R files need to be run first.
# DIAGNOSTIC ONLY: same fold-1 split as 29-07_15 (seed 2024, so fold
# membership is identical), R bumped 50 -> 200 to check if 29-07_15's
# fold-1 log-loss (1.268779, ~0.09 worse than clogit's ~1.18) was a
# simulation-noise artifact. Capped here -- if this doesn't close most
# of the gap, close the rpar lever rather than chase further (mirrors
# gmnl's Gotcha #24 discipline).

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

# ---- 2. Same seed-2024 fold split as 29-07_15 (identical fold 1 membership) ----
unique_cases <- unique(dt$Case)
case_folds <- data.table(
  Case = unique_cases,
  fold = sample(rep(1:5, length.out = length(unique_cases)))
)
dt <- merge(dt, case_folds, by = "Case", sort = FALSE)

train_wide <- dt[fold != 1]
test_wide  <- dt[fold == 1]
cat("train_wide:", nrow(train_wide), "rows | test_wide:", nrow(test_wide), "rows\n")

ml_train <- mlogit.data(
  as.data.frame(train_wide), shape = "wide", choice = "choice_col",
  varying = which(names(train_wide) %in% varying_cols), sep = "", id.var = "Case"
)
ml_test <- mlogit.data(
  as.data.frame(test_wide), shape = "wide", choice = "choice_col",
  varying = which(names(test_wide) %in% varying_cols), sep = "", id.var = "Case"
)

f <- mFormula(as.formula(paste("choice_col ~", paste(identified_vars, collapse = " + "), "| 0")))

cat("\n=== Fitting fold 1, R=200 (was R=50) — timing below ===\n")
fit_time <- system.time({
  fit_r200 <- tryCatch(
    mlogit(f, data = ml_train, rpar = c(Price = "n"), R = 200,
           halton = NA, panel = TRUE, print.level = 1),
    error = function(e) e
  )
})

if (inherits(fit_r200, "error")) {
  cat("FIT FAILED:", conditionMessage(fit_r200), "\n")
} else {
  cat("\nFit succeeded. Elapsed time:\n"); print(fit_time)
  print(summary(fit_r200))

  pred_test <- predict(fit_r200, newdata = ml_test)

  mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
    pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
    -mean(rowSums(actual_mat * log(pred_mat)))
  }

  df_test <- as.data.frame(ml_test)
  chid_order <- unique(df_test$chid)
  stopifnot(length(chid_order) == nrow(pred_test))
  lookup <- unique(df_test[, c("chid","Case","Task")])
  lookup <- lookup[match(chid_order, lookup$chid), ]

  actual_lookup <- test_wide[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]
  aligned <- merge(data.table(Case = lookup$Case, Task = lookup$Task),
                    actual_lookup, by = c("Case","Task"), sort = FALSE)
  stopifnot(nrow(aligned) == nrow(pred_test))

  actual_mat <- as.matrix(aligned[, .(Ch1, Ch2, Ch3, Ch4)])
  fold1_loss_r200 <- mlogloss(actual_mat, pred_test)
  cat("\nFold 1 log-loss, R=200:", fold1_loss_r200, "\n")
  cat("vs R=50 (29-07_15):      1.268779\n")
  cat("vs clogit benchmark:     ~1.18 mean (sd 0.009)\n")
}
