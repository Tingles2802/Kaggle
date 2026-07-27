# 28-07_2_oof_latent_class.R
# REQUIRES: none (fully self-contained, rebuilds its own data/fold exactly
# as 27-07_12 does — same seed/method, confirmed to match train$fold from
# 27-07_1). Run in its own R session, separate from 28-07_1 (this script
# does rm(list=ls()) and rebuilds long-format data differently).
#
# Purpose: 27-07_12 only printed fold log-losses for the latent-class MNL
# (Model 13), it never persisted predictions. This reruns the same Q=2 fit
# per fold and saves OOF predictions keyed by Case/Task, for Day 6
# ensembling. compute_lc_predictions() is unchanged/still validated (see
# 27-07_12 header).

rm(list = ls())

library(data.table)
library(gmnl)
library(mlogit)
library(Formula)

set.seed(2024)

# ---- 1. Load ----
dt <- fread("train.csv")

cat_attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP",
               "KA","SC","TS","NV","MA","LB","AF","HU")
identified_vars <- c(cat_attrs, "Price")

dt[, choice_col := fifelse(Ch1 == 1, "1",
                           fifelse(Ch2 == 1, "2",
                                   fifelse(Ch3 == 1, "3", "4")))]

varying_cols <- unlist(lapply(identified_vars, function(v) paste0(v, 1:4)))
stopifnot(all(varying_cols %in% names(dt)))

cases <- unique(dt$Case)
set.seed(2024)
folds <- sample(rep(1:5, length.out = length(cases)))
case_fold <- setNames(folds, cases)
dt[, fold := case_fold[as.character(Case)]]

mlogloss <- function(actual_mat, pred_mat) {
  eps <- 1e-15
  pred_mat <- pmin(pmax(pred_mat, eps), 1 - eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

# ---- 2. Manual LC prediction (gmnl has no predict() method) -- VALIDATED, see 27-07_12 header ----
compute_lc_predictions <- function(fit, mlogit_data_obj, identified_vars) {
  chid <- mlogit_data_obj$chid  # plain columns in this dfidx version
  alt  <- mlogit_data_obj$alt
  
  df <- as.data.frame(mlogit_data_obj)
  X  <- as.matrix(df[, identified_vars, drop = FALSE])
  
  cf <- coef(fit)
  beta1 <- cf[paste0("class.1.", identified_vars)]
  beta2 <- cf[paste0("class.2.", identified_vars)]
  
  U1 <- as.vector(X %*% beta1)
  U2 <- as.vector(X %*% beta2)
  
  build_prob_matrix <- function(U) {
    alt_chr  <- as.character(alt)
    chid_chr <- as.character(chid)
    u_chid   <- unique(chid_chr)
    alt_levels <- as.character(1:4)
    
    Umat <- sapply(alt_levels, function(a) {
      sel <- alt_chr == a
      v <- setNames(U[sel], chid_chr[sel])
      v[u_chid]
    })
    rownames(Umat) <- u_chid
    colnames(Umat) <- alt_levels
    
    expU <- exp(Umat - apply(Umat, 1, max))
    P <- expU / rowSums(expU)
    list(P = P, chid = u_chid)
  }
  
  res1 <- build_prob_matrix(U1)
  res2 <- build_prob_matrix(U2)
  stopifnot(identical(res1$chid, res2$chid))
  
  gamma_names <- grep("^\\(class\\)", names(cf), value = TRUE)
  gammas <- c(0, cf[gamma_names])
  w <- exp(gammas) / sum(exp(gammas))
  
  P_final <- w[1] * res1$P + w[2] * res2$P
  ord <- order(as.numeric(res1$chid))
  P_final <- P_final[ord, , drop = FALSE]
  chid_sorted <- res1$chid[ord]
  
  list(P = P_final, chid = chid_sorted, weights = w)
}

# ---- 3. lc formula: genuine 5-part, class-membership in rhs=5 ----
utility_part <- as.formula(paste("choice_col ~", paste(identified_vars, collapse = " + ")))
lc_formula <- mFormula(as.Formula(utility_part, ~ 0, ~ 0, ~ 0, ~ 1))
stopifnot(length(lc_formula)[2] == 5)

# ---- 4. Full 5-fold CV, Q=2, saving OOF predictions this time ----
Q <- 2
fold_scores <- numeric(5)
oof_lc <- vector("list", 5)

for (f in 1:5) {
  cat(sprintf("\n--- Fold %d ---\n", f))
  train_cases <- names(case_fold)[case_fold != f]
  test_cases  <- names(case_fold)[case_fold == f]
  
  train_wide <- dt[Case %in% as.integer(train_cases)]
  test_wide  <- dt[Case %in% as.integer(test_cases)]
  
  train_long <- mlogit.data(
    as.data.frame(train_wide), shape = "wide", choice = "choice_col",
    varying = which(names(train_wide) %in% varying_cols), sep = "", id.var = "Case"
  )
  test_long <- mlogit.data(
    as.data.frame(test_wide), shape = "wide", choice = "choice_col",
    varying = which(names(test_wide) %in% varying_cols), sep = "", id.var = "Case"
  )
  
  fit <- tryCatch(
    gmnl(lc_formula, data = train_long, model = "lc", Q = Q,
         panel = TRUE, method = "bfgs"),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    cat(sprintf("Fold %d: FIT FAILED — %s\n", f, conditionMessage(fit)))
    fold_scores[f] <- NA
    next
  }
  
  test_pred <- tryCatch(
    compute_lc_predictions(fit, test_long, identified_vars),
    error = function(e) e
  )
  if (inherits(test_pred, "error")) {
    cat(sprintf("Fold %d: PREDICTION FAILED — %s\n", f, conditionMessage(test_pred)))
    fold_scores[f] <- NA
    next
  }
  
  actual_mat <- as.matrix(test_wide[, .(Ch1, Ch2, Ch3, Ch4)])
  fold_scores[f] <- mlogloss(actual_mat, test_pred$P)
  cat(sprintf("Fold %d CV log-loss (Q=%d): %.6f | class weights: %s\n",
              f, Q, fold_scores[f], paste(round(test_pred$weights, 3), collapse = ", ")))
  
  # Recover Case/Task for each chid from test_long's own columns (confirmed
  # plain columns, see 27-07_12 header note #4) — don't assume chid's
  # numbering scheme, look it up directly.
  chid_lookup <- unique(as.data.table(as.data.frame(test_long))[, .(chid, Case, Task)])
  chid_lookup[, chid := as.character(chid)]
  fold_preds <- data.table(
    chid = test_pred$chid,
    lc13_Ch1 = test_pred$P[, "1"], lc13_Ch2 = test_pred$P[, "2"],
    lc13_Ch3 = test_pred$P[, "3"], lc13_Ch4 = test_pred$P[, "4"]
  )
  oof_lc[[f]] <- merge(chid_lookup, fold_preds, by = "chid", sort = FALSE)[
    , .(Case, Task, lc13_Ch1, lc13_Ch2, lc13_Ch3, lc13_Ch4)]
}

cat(sprintf("\n=== Q=%d mean CV: %.6f (sd %.6f) — expect ~1.235690 / 0.016979 ===\n",
            Q, mean(fold_scores, na.rm = TRUE), sd(fold_scores, na.rm = TRUE)))
print(fold_scores)

oof_lc <- rbindlist(oof_lc)
stopifnot(nrow(oof_lc) == nrow(dt))
fwrite(oof_lc, "oof_latent_class.csv")
cat("\nSaved oof_latent_class.csv (", nrow(oof_lc), "rows)\n")
