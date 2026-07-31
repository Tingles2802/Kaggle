# Standalone. No prereqs. Fresh session recommended (rm(list=ls()) first).
# Purpose: check for train/test distribution mismatches that could explain
# the stable ~0.047 CV/LB gap seen across all 8 submitted models.

library(data.table)

train <- fread("train.csv")
test  <- fread("test.csv")

attr_cols <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP",
               "KA","SC","TS","NV","MA","LB","AF","HU")
demo_cat  <- c("segment","miles","night","ppark","gender","age","educ",
               "region","Urb","income")

cat("=== Case/row counts ===\n")
cat("train cases:", uniqueN(train$Case), " rows:", nrow(train), "\n")
cat("test  cases:", uniqueN(test$Case),  " rows:", nrow(test),  "\n")

cat("\n=== Demographic categorical coverage (train vs test) ===\n")
for (v in demo_cat) {
  tr_lv <- sort(unique(train[[v]]))
  te_lv <- sort(unique(test[[v]]))
  missing_in_train <- setdiff(te_lv, tr_lv)
  missing_in_test   <- setdiff(tr_lv, te_lv)
  if (length(missing_in_train) > 0 || length(missing_in_test) > 0) {
    cat(sprintf("[%s] MISMATCH — in test not train: %s | in train not test: %s\n",
                v, paste(missing_in_train, collapse=","), paste(missing_in_test, collapse=",")))
  } else {
    cat(sprintf("[%s] OK — same %d levels\n", v, length(tr_lv)))
  }
}

cat("\n=== Demographic categorical PROPORTIONS (train vs test) ===\n")
for (v in demo_cat) {
  tr_p <- prop.table(table(train[[v]]))
  te_p <- prop.table(table(test[[v]]))
  cat(sprintf("--- %s ---\n", v))
  print(round(rbind(train=tr_p[names(tr_p)], test=te_p[names(tr_p)]), 3))
}

cat("\n=== Alt-specific attribute level coverage (per alt, per attribute) ===\n")
for (a in attr_cols) {
  for (alt in 1:4) {
    col <- paste0(a, alt)
    tr_lv <- sort(unique(train[[col]]))
    te_lv <- sort(unique(test[[col]]))
    if (!identical(tr_lv, te_lv)) {
      cat(sprintf("[%s] MISMATCH — train levels: %s | test levels: %s\n",
                  col, paste(tr_lv, collapse=","), paste(te_lv, collapse=",")))
    }
  }
}
cat("(only mismatches printed above; silence = full coverage match for all attr columns)\n")

cat("\n=== Price distributions per alt (train vs test) ===\n")
for (alt in 1:4) {
  col <- paste0("Price", alt)
  cat(sprintf("--- %s ---\n", col))
  cat("train:", paste(round(summary(train[[col]]), 1), collapse=" | "), "\n")
  cat("test :", paste(round(summary(test[[col]]),  1), collapse=" | "), "\n")
}

cat("\n=== Alt4 (structural opt-out) share ===\n")
cat("train Ch4 share:", mean(train$Ch4), "\n")
# test Ch1-4 are unlabeled targets (all 0.25 placeholders per sample_submission
# format) — do NOT compute a real share from test$Ch4, it's not the true label.
cat("test Ch4 column is a placeholder, not a real label — skipped.\n")

cat("\n=== Numeric demo aggregates (agea, milesa, nighta, incomea) ===\n")
for (v in c("agea","milesa","nighta","incomea")) {
  cat(sprintf("[%s] train mean/sd: %.2f/%.2f | test mean/sd: %.2f/%.2f\n",
              v, mean(train[[v]]), sd(train[[v]]), mean(test[[v]]), sd(test[[v]])))
}

cat("\n=== Tasks per case (train vs test) ===\n")
tr_tasks <- train[, .N, by=Case]$N
te_tasks <- test[, .N, by=Case]$N
cat("train tasks/case: min", min(tr_tasks), "max", max(tr_tasks), "mean", round(mean(tr_tasks),2), "\n")
cat("test  tasks/case: min", min(te_tasks), "max", max(te_tasks), "mean", round(mean(te_tasks),2), "\n")

cat("\n=== DONE — paste full console output back for review ===\n")
