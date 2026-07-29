# 29-07_14_rpar_oof_predict_check.R
# STANDALONE — no other R files need to be run first.
# DIAGNOSTIC ONLY (Section 1.2): confirms predict.mlogit() works on genuinely
# held-out chid's (new cases), not just in-sample data like 29-07_13 checked.
# Mirrors the clogit "new levels in strata(chid)" gotcha — checking rpar
# doesn't have the same failure mode before committing to a full CV script.

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

# ---- 2. Split into disjoint train_small (fit) / test_small (predict-only) ----
all_cases <- unique(dt$Case)
train_cases <- all_cases[1:200]
test_cases  <- all_cases[201:250]   # genuinely unseen cases, not in fit data

dt_train <- dt[Case %in% train_cases]
dt_test  <- dt[Case %in% test_cases]

ml_train <- mlogit.data(
  as.data.frame(dt_train), shape = "wide", choice = "choice_col",
  varying = which(names(dt_train) %in% varying_cols), sep = "", id.var = "Case"
)
ml_test <- mlogit.data(
  as.data.frame(dt_test), shape = "wide", choice = "choice_col",
  varying = which(names(dt_test) %in% varying_cols), sep = "", id.var = "Case"
)

# ---- 3. Fit on train_small only ----
f <- mFormula(as.formula(paste("choice_col ~", paste(identified_vars, collapse = " + "), "| 0")))

fit_diag <- tryCatch(
  mlogit(f, data = ml_train, rpar = c(Price = "n"), R = 50,
         halton = NA, panel = TRUE, print.level = 0),
  error = function(e) e
)

if (inherits(fit_diag, "error")) {
  cat("FIT FAILED:", conditionMessage(fit_diag), "\n")
} else {
  cat("Fit succeeded on train_small (", length(train_cases), "cases).\n")

  # ---- 4. predict() on genuinely out-of-fold data ----
  pred_oof <- tryCatch(predict(fit_diag, newdata = ml_test), error = function(e) e)

  if (inherits(pred_oof, "error")) {
    cat("OOF predict() FAILED:", conditionMessage(pred_oof), "\n")
  } else {
    cat("OOF predict() succeeded. class:", class(pred_oof), " dim:", paste(dim(pred_oof), collapse="x"), "\n")
    cat("Expected dim: ", length(test_cases)*4, "x4 (", length(test_cases), "cases x 4 tasks x 4 alts / rows, verify below)\n")
    print(head(pred_oof))
    cat("Row sums (should be ~1 each):\n")
    print(head(rowSums(pred_oof)))
    cat("Any NA in predictions:", anyNA(pred_oof), "\n")
  }
}
