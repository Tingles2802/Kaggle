# 29-07_12_rpar_diagnostic.R
# STANDALONE — no other R files need to be run first.
# DIAGNOSTIC ONLY (Section 1.2): small slice, inspect object structure before
# any full 5-fold CV run. mlogit::rpar is a distinct codepath from gmnl —
# not affected by gmnl's closure (Gotcha #22/#24).
# Paste in: fresh R console (own script, no prior sourcing required).

library(data.table)
library(mlogit)

set.seed(2024)

# ---- 1. Load + identify vars (hardcoded, matches 27-07_1's identified_vars) ----
dt <- fread("train.csv")

cat_attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP",
               "KA","SC","TS","NV","MA","LB","AF","HU")
identified_vars <- c(cat_attrs, "Price")

dt[, choice_col := fifelse(Ch1 == 1, "1",
                           fifelse(Ch2 == 1, "2",
                                   fifelse(Ch3 == 1, "3", "4")))]

varying_cols <- unlist(lapply(identified_vars, function(v) paste0(v, 1:4)))
stopifnot(all(varying_cols %in% names(dt)))

# ---- 2. Small slice (200 cases, mirrors 29-07_5's gmnl diagnostic size) ----
small_cases <- unique(dt$Case)[1:200]
dt_small <- dt[Case %in% small_cases]

ml_small <- mlogit.data(
  as.data.frame(dt_small), shape = "wide", choice = "choice_col",
  varying = which(names(dt_small) %in% varying_cols), sep = "", id.var = "Case"
)

# ---- 3. rpar fit: random coefficient on Price only (normal), rest fixed ----
# Formula shape: utility | 0 (no ASCs, matches clogit/gmnl conventions used so far)
f <- mFormula(as.formula(paste("choice_col ~", paste(identified_vars, collapse = " + "), "| 0")))

fit_diag <- tryCatch(
  mlogit(f, data = ml_small, rpar = c(Price = "n"), R = 50,
         halton = NA, panel = TRUE, print.level = 1),
  error = function(e) e
)

# ---- 4. Paste back everything below before proceeding ----
if (inherits(fit_diag, "error")) {
  cat("FIT FAILED:", conditionMessage(fit_diag), "\n")
} else {
  cat("packageVersion(mlogit):", as.character(packageVersion("mlogit")), "\n")
  print(class(fit_diag))
  print(names(fit_diag))
  print(summary(fit_diag))
  # Check whether a predict() method exists and works (unlike gmnl, Gotcha #22)
  pred_test <- tryCatch(predict(fit_diag, newdata = ml_small), error = function(e) e)
  if (inherits(pred_test, "error")) {
    cat("predict() FAILED:", conditionMessage(pred_test), "\n")
  } else {
    cat("predict() succeeded. class:", class(pred_test), " dim:", paste(dim(pred_test), collapse="x"), "\n")
    print(head(pred_test))
  }
}
