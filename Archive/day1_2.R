# =============================================================================
# Day 2 — Baseline model families
# Models:
#   A) nnet::multinom  — row-wise multinomial logit, demographics only,
#                        IGNORES alternative-specific attribute structure
#   B) mlogit           — long-format conditional logit, uses Price + the
#                        19 alt-specific feature columns per bundle
# CV scheme: Case-grouped 5-fold, seed = 2024 (MUST match Day 1 exactly for
#            scores to be comparable in the Model Registry)
# =============================================================================

# ---- 0. Packages -------------------------------------------------------
need <- c("data.table", "nnet", "mlogit", "dfidx")
for (pkg in need) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing package: ", pkg)
    install.packages(pkg)
  }
}
library(data.table)
library(nnet)
library(mlogit)
library(dfidx)

# ---- 1. Load data --------------------------------------------------------
# Reminder from Day 1 gotchas: check getwd()/list.files() first if this errors
# with "object 'train' not found" style issues.
train <- fread("train.csv")

# ---- 2. Case-grouped 5-fold CV assignment (must match Day 1) ------------
set.seed(2024)
unique_cases <- unique(train$Case)
case_folds <- data.table(
  Case = unique_cases,
  fold = sample(rep(1:5, length.out = length(unique_cases)))
)
train <- merge(train, case_folds, by = "Case", sort = FALSE)

# ---- 3. Log-loss function (must match Day 1 exactly) --------------------
mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

# Chosen-alternative index (1-4) from one-hot Ch1..Ch4 — used by both models
train[, chosen_alt := max.col(as.matrix(.SD)), .SDcols = c("Ch1","Ch2","Ch3","Ch4")]

# =============================================================================
# MODEL A: nnet::multinom — demographics only, no alt-specific attributes
# =============================================================================
factor_covariates <- c("segment", "year", "ppark", "gender", "educ", "region", "Urb")

train[, (factor_covariates) := lapply(.SD, factor), .SDcols = factor_covariates]

demo_formula <- as.formula(
  chosen_alt ~ segment + year + ppark + gender + educ + region + Urb +
    agea + milesa + nighta + incomea
)

align_new_levels <- function(tr, va, factor_cols) {
  for (col in factor_cols) {
    train_levels <- levels(factor(tr[[col]]))
    va_col <- as.character(va[[col]])
    unseen <- !(va_col %in% train_levels)
    if (any(unseen)) {
      mode_level <- names(sort(table(tr[[col]]), decreasing = TRUE))[1]
      cat("Note:", col, "-", sum(unseen), "unseen level row(s) in this fold, remapped to modal level:", mode_level, "\n")
      va_col[unseen] <- mode_level
    }
    va[[col]] <- factor(va_col, levels = train_levels)
  }
  va
}

fold_loss_A <- numeric(5)
for (i in 1:5) {
  tr <- train[fold != i]
  va <- train[fold == i]
  
  tr[, chosen_alt := factor(chosen_alt, levels = 1:4)]
  va <- align_new_levels(tr, va, factor_covariates)
  
  fit_A <- multinom(demo_formula, data = tr, trace = FALSE, maxit = 200)
  
  preds_A <- predict(fit_A, newdata = va, type = "probs")
  if (is.null(dim(preds_A))) {
    stop("Model A: predict() returned a vector, not a matrix — check that ",
         "all 4 classes appear in this fold's training data (fold ", i, ")")
  }
  preds_A <- preds_A[, as.character(1:4)]
  
  actual <- as.matrix(va[, .(Ch1, Ch2, Ch3, Ch4)])
  fold_loss_A[i] <- mlogloss(actual, preds_A)
  
  cat("Model A fold", i, "log-loss:", fold_loss_A[i], "\n")
}
cat("\nModel A (multinom, demographics only) — CV summary\n")
cat("Mean:", mean(fold_loss_A), " SD:", sd(fold_loss_A), "\n")
cat("Per-fold:", paste(round(fold_loss_A, 5), collapse = ", "), "\n\n")

# =============================================================================
# MODEL B: mlogit — long-format conditional logit, alt-specific attributes
# =============================================================================
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

cat("Price scale before standardization:\n")
print(summary(train_long$Price))
train_long[, Price := as.numeric(scale(Price))]

# Attach fold to long data via chid -> Case lookup
train_long[, Case := as.integer(sub("_.*", "", chid))]
train_long <- merge(train_long, case_folds, by = "Case", sort = FALSE)

# Diagnostic: for each attribute, what fraction of choice occasions (chid)
# show ANY variation across the 4 alternatives?
frac_varying <- sapply(attr_vars, function(v) {
  mean(train_long[, length(unique(get(v))), by = chid]$V1 > 1)
})
cat("Fraction of choice occasions where each attribute varies across alternatives:\n")
print(round(frac_varying, 4))

candidate_vars <- names(frac_varying)[frac_varying > 0]

rank_check_matrix <- function(X) {
  qr_x <- qr(X)
  rank <- qr_x$rank
  ncols <- ncol(X)
  redundant_cols <- if (rank < ncols) colnames(X)[qr_x$pivot[(rank + 1):ncols]] else character(0)
  list(rank = rank, ncol = ncols, redundant_cols = redundant_cols)
}

rank_check <- function(vars, data) {
  form <- as.formula(paste("~", paste(vars, collapse = " + ")))
  X <- model.matrix(form, data = data)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  rank_check_matrix(X)
}

attr_prefix <- function(colname, vars) {
  matches <- vars[sapply(vars, function(a) startsWith(colname, a))]
  if (length(matches) == 0) return(NA_character_)
  matches[which.max(nchar(matches))]
}

rc <- rank_check(candidate_vars, train_long)
cat("Row-level design matrix rank:", rc$rank, "of", rc$ncol, "columns\n")

identified_vars <- candidate_vars
if (length(rc$redundant_cols) > 0) {
  cat("Exact linearly-dependent columns found (row-level):\n")
  print(rc$redundant_cols)
  redundant_attrs <- unique(vapply(rc$redundant_cols, attr_prefix, character(1), vars = candidate_vars))
  redundant_attrs <- redundant_attrs[!is.na(redundant_attrs)]
  cat("Mapped back to attribute(s):", paste(redundant_attrs, collapse = ", "), "\n")
  identified_vars <- setdiff(candidate_vars, redundant_attrs)
}

within_chid_rank_check <- function(vars, data) {
  form <- as.formula(paste("~", paste(vars, collapse = " + ")))
  X <- model.matrix(form, data = data)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  col_names <- colnames(X)
  Xdt <- as.data.table(X)
  Xdt[, chid := data$chid]
  Xdm <- Xdt[, lapply(.SD, function(col) col - mean(col)), by = chid, .SDcols = col_names]
  Xdm_mat <- as.matrix(Xdm[, ..col_names])
  rank_check_matrix(Xdm_mat)
}

wc <- within_chid_rank_check(identified_vars, train_long)
cat("Within-chid (task) demeaned design matrix rank:", wc$rank, "of", wc$ncol, "columns\n")

if (length(wc$redundant_cols) > 0) {
  cat("Exact linearly-dependent columns found (within-task):\n")
  print(wc$redundant_cols)
  redundant_attrs2 <- unique(vapply(wc$redundant_cols, attr_prefix, character(1), vars = identified_vars))
  redundant_attrs2 <- redundant_attrs2[!is.na(redundant_attrs2)]
  cat("Mapped back to attribute(s):", paste(redundant_attrs2, collapse = ", "), "\n")
  cat("Dropping these from the model formula — they're unidentifiable within",
      "conditional logit's within-task likelihood even though they vary",
      "across the full dataset.\n")
  identified_vars <- setdiff(identified_vars, redundant_attrs2)
}

mlogit_formula <- as.formula(
  paste0("chosen ~ ", paste(identified_vars, collapse = " + "), " | 1")
)

chid_check <- train_long[, .(n_alts = .N, n_chosen = sum(chosen)), by = chid]
cat("Choice-set size distribution (should be ALL 4):\n")
print(table(chid_check$n_alts))
cat("Chosen-count distribution (should be ALL 1):\n")
print(table(chid_check$n_chosen))

full_idx <- dfidx(train_long, idx = list("chid", "alt"), choice = "chosen")
test_fit <- tryCatch(
  mlogit(chosen ~ CC | 0, data = full_idx),
  error = function(e) {
    cat("Minimal single-attribute (CC, no ASC) test model FAILED:", conditionMessage(e), "\n")
    NULL
  }
)
if (!is.null(test_fit)) {
  cat("Minimal single-attribute (CC, no ASC) test model SUCCEEDED — coefficients:\n")
  print(coef(test_fit))
}

fold_loss_B <- numeric(5)
for (i in 1:5) {
  tr_long <- train_long[fold != i]
  va_long <- train_long[fold == i]
  
  for (v in cat_attr_vars) {
    full_levels <- nlevels(train_long[[v]])
    fold_levels <- length(unique(tr_long[[v]]))
    if (fold_levels < full_levels) {
      cat("Note: fold", i, "-", v, "has", fold_levels, "of", full_levels,
          "levels present in training data (possible singularity source)\n")
    }
  }
  
  tr_idx <- dfidx(tr_long, idx = list("chid", "alt"), choice = "chosen")
  
  # FIX: the previous version's error handler called the no-ASC retry
  # directly (unwrapped). When that retry ALSO failed (as it did — same
  # "exactly singular" error), the resulting error had nothing left to
  # catch it, so it propagated out of the entire for loop and killed the
  # whole script — which is why nothing below fold 1 (including the
  # Diagnostic C/D output at the bottom of this file) ever ran. Nest a
  # second tryCatch around the retry so a double-failure degrades to a
  # skipped fold (fit_B <- NULL) instead of stopping execution.
  fit_B <- tryCatch(
    mlogit(mlogit_formula, data = tr_idx, method = "bfgs"),
    error = function(e) {
      cat("Fold", i, "- bfgs with ASCs failed:", conditionMessage(e), "\n")
      cat("Fold", i, "- retrying without alternative-specific constants (| 0)...\n")
      formula_no_asc <- update(mlogit_formula, . ~ . | 0)
      tryCatch(
        mlogit(formula_no_asc, data = tr_idx, method = "bfgs"),
        error = function(e2) {
          cat("Fold", i, "- no-ASC retry also failed:", conditionMessage(e2), "\n")
          NULL
        }
      )
    }
  )
  
  if (is.null(fit_B)) {
    cat("Fold", i, "- skipping (Model B failed to fit in both attempts)\n")
    fold_loss_B[i] <- NA
    next
  }
  
  va_idx <- dfidx(va_long, idx = list("chid", "alt"), choice = "chosen")
  preds_B_long <- predict(fit_B, newdata = va_idx)
  
  chid_order <- rownames(preds_B_long)
  va_wide_order <- unique(va_long[, .(chid, Case, Task)])
  setkey(va_wide_order, chid)
  va_wide_order <- va_wide_order[chid_order]
  
  actual_lookup <- train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]
  actual_aligned <- merge(va_wide_order, actual_lookup, by = c("Case","Task"), sort = FALSE)
  actual <- as.matrix(actual_aligned[, .(Ch1, Ch2, Ch3, Ch4)])
  
  fold_loss_B[i] <- mlogloss(actual, as.matrix(preds_B_long))
  
  cat("Model B fold", i, "log-loss:", fold_loss_B[i], "\n")
}
cat("\nModel B (mlogit, alt-specific attributes + ASCs) — CV summary\n")
cat("Mean:", mean(fold_loss_B, na.rm = TRUE), " SD:", sd(fold_loss_B, na.rm = TRUE),
    "(", sum(is.na(fold_loss_B)), "fold(s) skipped)\n")
cat("Per-fold:", paste(round(fold_loss_B, 5), collapse = ", "), "\n\n")

# =============================================================================
# Comparison printout — paste this whole block's console output back
# =============================================================================
cat("=== Day 2 CV Comparison ===\n")
cat("Uniform baseline (Day 1):           1.386294\n")
cat("Marginal-frequency (Day 1):         1.378770\n")
cat("Model A - multinom (demo only):    ", round(mean(fold_loss_A), 6), "\n")
cat("Model B - mlogit (alt attributes): ", round(mean(fold_loss_B, na.rm = TRUE), 6), "\n")


# --- Diagnostic C: level frequency check ---
attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP",
           "KA","SC","TS","NV","MA","LB","AF","HU")

level_counts <- lapply(attrs, function(a) {
  cols <- paste0(a, 1:4)
  vals <- unlist(train[, ..cols])
  table(vals)
})
names(level_counts) <- attrs
print(level_counts)
# Look for any level with a very small count (e.g. < 20) relative to others -
# a candidate for quasi-complete separation.

# --- Diagnostic D: bisection to find the breaking point ---
# FIX: previous version referenced `mlogit_data` and `Choice`, neither of
# which exist in this script (the actual dfidx object built above is
# `full_idx`, and the outcome column is `chosen`). Reuse full_idx directly —
# it already has Price scaled/numeric and all 19 attributes as factors.
fit_subset <- function(attr_subset) {
  fmla <- as.formula(paste("chosen ~ Price +",
                           paste(attr_subset, collapse = " + "), "| 0"))
  tryCatch({
    m <- mlogit(fmla, data = full_idx)
    "OK"
  }, error = function(e) conditionMessage(e))
}

# FIX: seq(2, length(attrs), by = 2) with length(attrs) == 19 produces
# 2,4,...,18 and NEVER reaches 19 -- every prior "OK" excluded HU (last in
# `attrs`) and never actually tested the full model that fails in the fold
# loop above. Iterate every k up to the true length instead.
results <- data.frame(n_attrs = integer(), attrs = character(), status = character())
for (k in 2:length(attrs)) {
  subset_k <- attrs[1:k]
  status <- fit_subset(subset_k)
  results <- rbind(results, data.frame(n_attrs = k,
                                       attrs = paste(subset_k, collapse=","),
                                       status = status))
  cat(sprintf("k=%d: %s\n", k, status))
}
print(results)

# If k=19 fails where k=18 succeeded, that implicates HU specifically --
# but it could also just be an artifact of reaching 19 attributes at all.
# Leave-one-out over the full set disambiguates: each test still uses 18
# attributes, just a different 18 each time.
loo_results <- data.frame(dropped = character(), status = character())
for (a in attrs) {
  subset_loo <- setdiff(attrs, a)
  status <- fit_subset(subset_loo)
  loo_results <- rbind(loo_results, data.frame(dropped = a, status = status))
  cat(sprintf("drop %s: %s\n", a, status))
}
print(loo_results)

# --- Diagnostic E: per-fold within-chid rank check ---
# The within-chid rank check above was only ever run ONCE, on the full
# dataset (61/61, fine). But Diagnostics C/D just showed the full 19-attr +
# Price model (no ASCs) fits with NO problem on the full data -- so the
# singularity must be specific to a FOLD'S TRAINING SUBSET (~80% of chids),
# not the attribute set itself. Re-run the same within-chid rank check
# per fold to find which column(s) lose identification once that fold's
# held-out chids are removed from training.
for (i in 1:5) {
  tr_long_i <- train_long[fold != i]
  wc_i <- within_chid_rank_check(identified_vars, tr_long_i)
  cat("Fold", i, "- within-chid rank:", wc_i$rank, "of", wc_i$ncol, "columns\n")
  if (length(wc_i$redundant_cols) > 0) {
    cat("  Redundant columns (fold", i, "training subset only):\n")
    print(wc_i$redundant_cols)
    redundant_attrs_i <- unique(vapply(wc_i$redundant_cols, attr_prefix, character(1), vars = identified_vars))
    cat("  Mapped to attribute(s):", paste(redundant_attrs_i[!is.na(redundant_attrs_i)], collapse = ", "), "\n")
  }
}