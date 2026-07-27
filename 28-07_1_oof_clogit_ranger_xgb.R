# 28-07_1_oof_clogit_ranger_xgb.R
# REQUIRES (run first, in order): 27-07_1_pipeline_foundation.R,
#   27-07_4_trees_setup.R (Steps 1-2), 27-07_5_xgb_onehot.R,
#   27-07_10_clogit_simplified_segment.R, 27-07_13_demo_price_interactions_screen.R
#   (this also runs 27-07_13's own one-shot Model-14 screen as a side effect
#   of sourcing it — harmless, ignore that printed output).
#
# Purpose: generate out-of-fold (OOF) predictions from the three current best
# tuned models — clogit Model 15, ranger Model 18, xgboost Model 20 (winner
# of the depth-extension grid) — for Day 6 ensembling. Saves one combined CSV
# keyed by Case/Task. Latent-class MNL (Model 13) is handled separately in
# 28-07_2 since it needs its own environment (rm(list=ls()), long-format
# rebuild) — do not run in the same session as this script.

library(survival)
library(ranger)
library(xgboost)
library(data.table)

model15_formula <- as.formula(paste0(
  "chosen ~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender + strata(chid)"
))
model15_score_formula <- as.formula(paste0(
  "~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender"
))

oof_clogit <- vector("list", 5)
oof_ranger <- vector("list", 5)
oof_xgb    <- vector("list", 5)

for (f in 1:5) {
  cat("=== Fold", f, "===\n")
  
  ## ---- clogit, Model 15 ----
  tr_long_f <- train_long_ext[fold != f]
  va_long_f <- copy(train_long_ext[fold == f])
  
  fit_c <- clogit(model15_formula, data = tr_long_f, method = "exact")
  mm <- model.matrix(model15_score_formula, data = va_long_f)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  keep_coef <- names(coef(fit_c))[!is.na(coef(fit_c))]
  mm <- mm[, keep_coef, drop = FALSE]
  va_long_f[, score := as.numeric(mm %*% coef(fit_c)[keep_coef])]
  va_long_f[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]
  
  setorder(va_long_f, chid, alt)
  preds_c <- dcast(va_long_f, chid ~ alt, value.var = "prob")
  setnames(preds_c, as.character(1:4), c("clogit15_Ch1","clogit15_Ch2","clogit15_Ch3","clogit15_Ch4"))
  ct_key <- unique(va_long_f[, .(chid, Case, Task)])
  setkey(ct_key, chid)
  preds_c <- merge(ct_key, preds_c, by = "chid", sort = FALSE)
  oof_clogit[[f]] <- preds_c[, .(Case, Task, clogit15_Ch1, clogit15_Ch2, clogit15_Ch3, clogit15_Ch4)]
  
  ## ---- ranger, Model 18 (mtry=40, min.node.size=60) ----
  tr_r <- ranger_data[fold != f]
  va_r <- ranger_data[fold == f]
  fit_r <- ranger(y ~ ., data = tr_r[, !c("fold"), with = FALSE], probability = TRUE,
                  num.trees = 500, mtry = 40, min.node.size = 60, seed = 2024)
  preds_r <- predict(fit_r, data = va_r)$predictions
  colnames(preds_r) <- c("ranger18_Ch1","ranger18_Ch2","ranger18_Ch3","ranger18_Ch4")
  oof_ranger[[f]] <- data.table(Case = train[fold == f, Case], Task = train[fold == f, Task], preds_r)
  
  ## ---- xgboost, Model 20 (eta=0.1, depth=3) ----
  tr_idx <- which(train$fold != f)
  va_idx <- which(train$fold == f)
  dtrain <- xgb.DMatrix(data = xgb_data_oh[tr_idx, ], label = y_zero_indexed[tr_idx])
  dtest  <- xgb.DMatrix(data = xgb_data_oh[va_idx, ], label = y_zero_indexed[va_idx])
  fit_x <- xgb.train(
    params = list(objective = "multi:softprob", num_class = 4,
                  eval_metric = "mlogloss", eta = 0.1, max_depth = 3),
    data = dtrain, nrounds = 1000, evals = list(val = dtest),
    early_stopping_rounds = 20, verbose = 0
  )
  best_iter <- as.integer(xgb.attr(fit_x, "best_iteration")) + 1L
  preds_x <- predict(fit_x, dtest, iterationrange = c(1L, best_iter))
  colnames(preds_x) <- c("xgb20_Ch1","xgb20_Ch2","xgb20_Ch3","xgb20_Ch4")
  oof_xgb[[f]] <- data.table(Case = train[va_idx, Case], Task = train[va_idx, Task], preds_x)
}

oof_clogit <- rbindlist(oof_clogit)
oof_ranger <- rbindlist(oof_ranger)
oof_xgb    <- rbindlist(oof_xgb)

oof_all <- merge(train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)], oof_clogit, by = c("Case","Task"), sort = FALSE)
oof_all <- merge(oof_all, oof_ranger, by = c("Case","Task"), sort = FALSE)
oof_all <- merge(oof_all, oof_xgb, by = c("Case","Task"), sort = FALSE)
stopifnot(nrow(oof_all) == nrow(train))

# Pipeline-refactor rule: reproduce each model's already-logged CV score
# from its OOF matrix before trusting this file for ensembling.
actual_mat <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
cat("\n--- verification vs logged CV scores ---\n")
cat("clogit15 (expect ~1.162469):", mlogloss(actual_mat, as.matrix(oof_all[, .(clogit15_Ch1,clogit15_Ch2,clogit15_Ch3,clogit15_Ch4)])), "\n")
cat("ranger18 (expect ~1.189566):", mlogloss(actual_mat, as.matrix(oof_all[, .(ranger18_Ch1,ranger18_Ch2,ranger18_Ch3,ranger18_Ch4)])), "\n")
cat("xgb20    (expect ~1.176378):", mlogloss(actual_mat, as.matrix(oof_all[, .(xgb20_Ch1,xgb20_Ch2,xgb20_Ch3,xgb20_Ch4)])), "\n")

fwrite(oof_all, "oof_clogit_ranger_xgb.csv")
cat("\nSaved oof_clogit_ranger_xgb.csv (", nrow(oof_all), "rows)\n")