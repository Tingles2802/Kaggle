# =============================================================================
# Pipeline Foundation — replaces the setup portions of day1_2.R
#
# WHY THIS EXISTS: day1_2.R mixed one-time setup (data load, folds, long-format
# build) together with a lot of now-dead work: Model A (multinom, already
# confirmed at 1.370529 — no need to refit), the abandoned mlogit fits (5 folds
# x 2 retries), and Diagnostics C/D/E (~36 extra mlogit fits + per-fold rank
# checks). All of that is resolved and logged (see experiment_log Section 5) —
# rerunning it costs real time and gives no new information.
#
# This script keeps ONLY what every future model (clogit, trees, latent-class)
# needs: loaded data, the exact Case-grouped 5-fold split (seed 2024, must stay
# identical for CV comparability), the log-loss function, and train_long in
# analysis-ready form. `identified_vars` is hardcoded below from Day 2's rank
# checks rather than recomputed — those checks are deterministic given the
# data and already came back full-rank at every stage (row-level, within-chid,
# and per-fold), so nothing was actually dropped.
#
# Run this ONE script, then build Day 3 work (interactions, ranger, xgboost)
# on top of the objects it produces: train, train_long, case_folds, mlogloss(),
# attr_vars, identified_vars.
# =============================================================================

# ---- 0. Packages ----------------------------------------------------------
# Only data.table needed here. survival (for clogit) loaded in the model
# script that consumes this. nnet/mlogit/dfidx deliberately NOT loaded —
# not needed unless/until the mixed-logit stretch goal is revisited.
need <- c("data.table")
for (pkg in need) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing package: ", pkg)
    install.packages(pkg)
  }
}
library(data.table)

# ---- 1. Load data -----------------------------------------------------------
train <- fread("train.csv")

# ---- 2. Case-grouped 5-fold CV assignment (must match Day 1/2 exactly) -----
set.seed(2024)
unique_cases <- unique(train$Case)
case_folds <- data.table(
  Case = unique_cases,
  fold = sample(rep(1:5, length.out = length(unique_cases)))
)
train <- merge(train, case_folds, by = "Case", sort = FALSE)

# ---- 3. Log-loss function (must match Day 1 exactly) -----------------------
mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

# Chosen-alternative index (1-4) — kept for any row-wise model (e.g. trees).
train[, chosen_alt := max.col(as.matrix(.SD)), .SDcols = c("Ch1","Ch2","Ch3","Ch4")]

# ---- 4. Long-format build (alt-specific attributes + Price) ---------------
attr_vars <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP",
               "KA","SC","TS","NV","MA","LB","AF","HU","Price")

build_long <- function(dt, alt) {
  cols <- paste0(attr_vars, alt)
  out <- dt[, ..cols]
  setnames(out, cols, attr_vars)
  out[, `:=`(
    Case = dt$Case,
    Task = dt$Task,
    chid = paste0(dt$Case, "_", dt$Task),
    alt  = alt,
    chosen = dt[[paste0("Ch", alt)]] == 1
  )]
  out
}

train_long <- rbindlist(lapply(1:4, build_long, dt = train))
setorder(train_long, chid, alt)

cat_attr_vars <- setdiff(attr_vars, "Price")
for (v in cat_attr_vars) train_long[, (v) := as.factor(get(v))]
train_long[, Price := as.numeric(Price)]
train_long[, Price := as.numeric(scale(Price))]

# Attach fold to long data via chid -> Case lookup
train_long[, Case := as.integer(sub("_.*", "", chid))]
train_long <- merge(train_long, case_folds, by = "Case", sort = FALSE)

# ---- 5. Identified variables (hardcoded — resolved in Day 2 diagnostics) --
# Day 2 confirmed via row-level rank check (61/61), within-chid rank check on
# full data (61/61), and within-chid rank check on every fold's training
# subset (61/61 in all 5 folds) that all 19 attributes are fully identified —
# nothing was dropped. See experiment_log Section 5 ("Things Tried and Ruled
# Out") before re-deriving this.
identified_vars <- cat_attr_vars  # all 19 attribute columns, Price handled separately

cat("=== Pipeline foundation ready ===\n")
cat("train:", nrow(train), "rows /", length(unique_cases), "cases\n")
cat("train_long:", nrow(train_long), "rows\n")
cat("identified_vars (", length(identified_vars), "):", paste(identified_vars, collapse=", "), "\n")
