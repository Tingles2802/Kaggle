# Prereqs: none. Self-contained — reads train.csv directly.
# Purpose: DIAGNOSTIC ONLY (Section 1.2 rule). Not a full model fit.
# Goal: check (a) mlogit::rpar converges on a small slice, (b) predict() method
# exists for the fitted object (gmnl's absence of predict.gmnl() was the failure
# mode we're checking isn't repeated here), (c) fit time on small N as a proxy
# for full-data feasibility. NOT for CV score / submission use.
# Paste in: console (run top to bottom, inspect each print() output manually).

# rm(list = ls())
library(data.table)
library(mlogit)

cat("mlogit version:", as.character(packageVersion("mlogit")), "\n")

dt <- fread("train.csv")

# --- small slice: first 50 unique Cases only (diagnostic speed, not full data) ---
slice_cases <- unique(dt$Case)[1:50]
dts <- dt[Case %in% slice_cases]
cat("Slice rows (tasks):", nrow(dts), " unique cases:", length(slice_cases), "\n")

# unique choice occasion id = Case-Task combo
dts[, chid := paste(Case, Task, sep = "_")]

attr_vars <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
               "PP","KA","SC","TS","NV","MA","LB","AF","HU","Price")

# reshape wide (Case x alt-suffixed cols) -> long (one row per chid-alt)
long_list <- list()
for (j in 1:4) {
  cols_j <- paste0(attr_vars, j)
  sub <- dts[, c("chid", "Case", cols_j, paste0("Ch", j)), with = FALSE]
  setnames(sub, cols_j, attr_vars)
  setnames(sub, paste0("Ch", j), "choice01")
  sub[, alt := j]
  long_list[[j]] <- sub
}
long <- rbindlist(long_list)
setorder(long, chid, alt)
long[, choice := as.logical(choice01)]
long[, choice01 := NULL]

# sanity: each chid has exactly one TRUE
chk <- long[, .(n_chosen = sum(choice)), by = chid]
cat("chids with != 1 chosen alt (should be 0):", sum(chk$n_chosen != 1), "\n")

mldata <- mlogit.data(long, choice = "choice", shape = "long",
                      chid.var = "chid", alt.var = "alt", id.var = "Case")

# --- diagnostic fit: Price as random coefficient only, small Halton draw count ---
# keeping spec minimal on purpose — this is a codepath/convergence check, not
# a real spec (full spec + Price random + other attrs comes later if this passes)
form <- mFormula(choice ~ Price | 0)

t0 <- Sys.time()
fit <- tryCatch(
  mlogit(form, data = mldata,
         rpar = c(Price = "n"),
         R = 50, halton = NA, panel = TRUE,
         method = "bfgs"),
  error = function(e) { cat("FIT ERROR:", conditionMessage(e), "\n"); NULL }
)
t1 <- Sys.time()
cat("Fit time:", as.numeric(difftime(t1, t0, units = "secs")), "sec\n")

if (!is.null(fit)) {
  cat("Converged (fit$convergence, 0 = yes):", fit$convergence, "\n")
  print(names(fit))
  print(summary(fit))
  
  # the key check from Section 4a: does predict() actually work (gmnl lacked this)
  pred_test <- tryCatch(predict(fit, newdata = mldata), error = function(e) {
    cat("PREDICT ERROR:", conditionMessage(e), "\n"); NULL
  })
  if (!is.null(pred_test)) {
    cat("predict() succeeded. class:", class(pred_test), " dim:",
        paste(dim(pred_test), collapse = "x"), "\n")
    print(head(pred_test))
  }
}