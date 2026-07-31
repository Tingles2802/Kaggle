# 31-07_3_panel_base_retrofit.R
# STANDALONE - no other R files required. Fresh fread of train.csv + the panel
# feature CSVs from 30-07_5_panel_features.R (run that first if those CSVs don't
# exist yet). Run in a CLEAN session (rm(list=ls()) recommended - Gotcha #32/#33
# class of issue: this script derives its own fold split, doesn't rely on train$fold).
#
# Purpose: test whether panel features improve BASE clogit/ranger/xgboost directly
# (not just the meta-learner, which already showed base-xgboost Delta=0.0034).
#
# Panel feature handling by model type:
#  - PriceDev1-3 are genuinely alt-specific (alt's own price minus Case baseline)
#    -> used like Price itself in all 3 models. PriceDev4 := 0, matching Price4's
#    structural-zero convention (Alt4 opt-out).
#  - case_mean_price, case_sd_price, TaskFrac, IsFirstTask are Case/Task-constant
#    (same across all 4 alts in a task) -> for clogit, only identified via Price:
#    interaction (mirrors Model 15's Price:agea_z etc.); for ranger/xgboost, used
#    directly as columns (no differencing issue for trees).
#
# ASSUMPTIONS (not verifiable from the 5 reference files - flag if base CV doesn't
# reproduce the "expect ~X" prints below):
#  - attr_vars = the 19 categorical attribute short-codes (CC..HU) + "Price" for
#    the alt-specific block, matching train.csv's column schema.
#  - Case-grouped 5-fold, seed 2024, rebuilt fresh here (exact original assignment
#    code was not in the 5 files provided).

library(data.table); library(survival); library(ranger); library(xgboost)

mlogloss <- function(actual_mat, pred_mat, eps = 1e-15) {
  pred_mat <- pmax(pmin(pred_mat, 1 - eps), eps)
  -mean(rowSums(actual_mat * log(pred_mat)))
}

# ---- Step 0: load data, merge panel features, build fold split ----
train <- fread("train.csv")
panel_train <- fread("case_panel_features_train.csv")
stopifnot(nrow(panel_train) == nrow(train))
train <- merge(train, panel_train, by = c("Case", "Task", "No"), sort = FALSE)
stopifnot(nrow(train) == 21565)

set.seed(2024)
case_ids <- unique(train$Case)
fold_map <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train <- merge(train, fold_map, by = "Case", sort = FALSE)

attr_vars <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")
demo_cols <- c("segmentind","yearind","pparkind","genderind","educind","regionind","Urbind",
               "agea","milesa","nighta","incomea")

train[, PriceDev4 := 0]  # structural zero, same convention as Price4

# ==========================================================================
# PART A: clogit (Model 15 spec) - base vs +panel
# ==========================================================================

build_long <- function(dt) {
  pieces <- lapply(1:4, function(a) {
    cols <- c(paste0(attr_vars, a), paste0("Price", a), paste0("PriceDev", a))
    sub <- dt[, ..cols]
    setnames(sub, c(attr_vars, "Price", "PriceDev"))
    sub[, `:=`(Case = dt$Case, Task = dt$Task, No = dt$No, alt = a,
               chosen = dt[[paste0("Ch", a)]], fold = dt$fold,
               case_mean_price = dt$case_mean_price, case_sd_price = dt$case_sd_price,
               TaskFrac = dt$TaskFrac, IsFirstTask = dt$IsFirstTask,
               segment = dt$segment, agea = dt$agea, Urbind = dt$Urbind, gender = dt$gender)]
    sub
  })
  out <- rbindlist(pieces)
  out[, chid := paste0(Case, "_", Task)]
  out[, agea_z := as.numeric(scale(agea))]
  out[, is_luxury_segment := as.numeric(grepl("Luxury", segment, ignore.case = TRUE))]
  out
}
train_long <- build_long(train)

# FIX (2026-07-31): 27-07_1 confirms attr_vars must be factor-coded (else clogit
# fits them as linear numeric, not categorical dummies) and Price must be
# z-scored (global scale over train_long). Both were missing in the first draft
# and caused clogit BASE to not reproduce (1.211016 vs expected ~1.162469).
for (v in attr_vars) train_long[, (v) := as.factor(get(v))]
train_long[, Price := as.numeric(scale(Price))]

rhs_vars <- c(attr_vars, "Price")
base_formula  <- as.formula(paste0("chosen ~ ", paste(rhs_vars, collapse = " + "),
                                   " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender + strata(chid)"))
base_score    <- as.formula(paste0("~ ", paste(rhs_vars, collapse = " + "),
                                   " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender"))
panel_formula <- as.formula(paste0("chosen ~ ", paste(rhs_vars, collapse = " + "),
                                   " + PriceDev + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
                                   " + Price:TaskFrac + Price:case_sd_price + strata(chid)"))
panel_score   <- as.formula(paste0("~ ", paste(rhs_vars, collapse = " + "),
                                   " + PriceDev + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
                                   " + Price:TaskFrac + Price:case_sd_price"))

score_clogit_cv <- function(fit_formula, score_formula, data_long, actual_full) {
  losses <- numeric(5)
  for (i in 1:5) {
    tr <- data_long[fold != i]
    va <- copy(data_long[fold == i])
    fit <- clogit(fit_formula, data = tr, method = "exact")
    mm <- model.matrix(score_formula, data = va)
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
    keep <- names(coef(fit))[!is.na(coef(fit))]
    mm <- mm[, keep, drop = FALSE]
    va[, score := as.numeric(mm %*% coef(fit)[keep])]
    va[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]
    setorder(va, chid, alt)
    pw <- dcast(va, chid ~ alt, value.var = "prob")
    setnames(pw, as.character(1:4), c("Ch1","Ch2","Ch3","Ch4"))
    key_i <- unique(va[, .(chid, Case, Task)]); setkey(key_i, chid)
    pw <- merge(key_i, pw, by = "chid", sort = FALSE)
    actual_i <- merge(pw[, .(Case, Task)], actual_full, by = c("Case","Task"), sort = FALSE)
    losses[i] <- mlogloss(as.matrix(actual_i[, .(Ch1,Ch2,Ch3,Ch4)]), as.matrix(pw[, .(Ch1,Ch2,Ch3,Ch4)]))
    cat("clogit fold", i, "loss:", losses[i], "\n")
  }
  losses
}

actual_lookup <- train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]

cv_clogit_base  <- score_clogit_cv(base_formula, base_score, train_long, actual_lookup)
cat("clogit BASE  - mean:", mean(cv_clogit_base), " sd:", sd(cv_clogit_base), " (expect ~1.162469)\n")

cv_clogit_panel <- score_clogit_cv(panel_formula, panel_score, train_long, actual_lookup)
cat("clogit PANEL - mean:", mean(cv_clogit_panel), " sd:", sd(cv_clogit_panel), "\n")
cat("Delta (base - panel):", mean(cv_clogit_base) - mean(cv_clogit_panel), "\n\n")

# ==========================================================================
# PART B: ranger (Model 18 spec) - base vs +panel
# ==========================================================================
cat_feature_cols <- unlist(lapply(1:4, function(a) paste0(attr_vars, a)))
price_cols <- paste0("Price", 1:4)
pricedev_cols <- paste0("PriceDev", 1:4)
panel_extra <- c("case_mean_price","case_sd_price","TaskFrac","IsFirstTask")

ranger_base_cols  <- c(cat_feature_cols, price_cols, demo_cols)
ranger_panel_cols <- c(ranger_base_cols, pricedev_cols, panel_extra)

build_ranger_data <- function(cols) {
  rd <- copy(train[, ..cols])
  for (v in cat_feature_cols) rd[, (v) := as.factor(get(v))]
  rd[, y := factor((train$Ch1==1)*1 + (train$Ch2==1)*2 + (train$Ch3==1)*3 + (train$Ch4==1)*4, levels = 1:4)]
  rd[, fold := train$fold]
  rd
}
ranger_data_base  <- build_ranger_data(ranger_base_cols)
ranger_data_panel <- build_ranger_data(ranger_panel_cols)

run_ranger_cv <- function(rd) {
  losses <- numeric(5)
  for (k in 1:5) {
    tr <- rd[fold != k]; va <- rd[fold == k]
    fit <- ranger(y ~ ., data = tr[, !c("fold"), with = FALSE], probability = TRUE,
                  num.trees = 500, mtry = 40, min.node.size = 60, seed = 2024)
    preds <- predict(fit, data = va)$predictions
    colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
    actual <- as.matrix(train[fold == k, .(Ch1,Ch2,Ch3,Ch4)])
    losses[k] <- mlogloss(actual, preds)
    cat("ranger fold", k, "loss:", losses[k], "\n")
  }
  losses
}
cv_ranger_base  <- run_ranger_cv(ranger_data_base)
cat("ranger BASE  - mean:", mean(cv_ranger_base), " (expect ~1.189566)\n")
cv_ranger_panel <- run_ranger_cv(ranger_data_panel)
cat("ranger PANEL - mean:", mean(cv_ranger_panel), "\n")
cat("Delta:", mean(cv_ranger_base) - mean(cv_ranger_panel), "\n\n")

# ==========================================================================
# PART C: xgboost one-hot (Model 20 spec) - base vs +panel
# ==========================================================================
build_onehot <- function(dt, cols) {
  mats <- lapply(cols, function(v) {
    x <- as.factor(dt[[v]]); lv <- levels(x)
    m <- sapply(lv, function(l) as.integer(x == l))
    if (length(lv) == 1) m <- matrix(m, ncol = 1)
    colnames(m) <- paste0(v, lv)
    m
  })
  do.call(cbind, mats)
}
onehot_mat    <- build_onehot(train, cat_feature_cols)
numeric_base  <- as.matrix(train[, c(price_cols, demo_cols), with = FALSE])
numeric_panel <- as.matrix(train[, c(price_cols, demo_cols, pricedev_cols, panel_extra), with = FALSE])

xgb_data_base  <- cbind(onehot_mat, numeric_base)
xgb_data_panel <- cbind(onehot_mat, numeric_panel)
y_idx <- (train$Ch1==1)*0 + (train$Ch2==1)*1 + (train$Ch3==1)*2 + (train$Ch4==1)*3

run_xgb_cv <- function(X) {
  losses <- numeric(5)
  for (k in 1:5) {
    tr <- which(train$fold != k); va <- which(train$fold == k)
    dtr <- xgb.DMatrix(X[tr, ], label = y_idx[tr])
    dva <- xgb.DMatrix(X[va, ], label = y_idx[va])
    fit <- xgb.train(params = list(objective = "multi:softprob", num_class = 4,
                                   eval_metric = "mlogloss", eta = 0.1, max_depth = 3),
                     data = dtr, nrounds = 1000, evals = list(val = dva),
                     early_stopping_rounds = 20, verbose = 0)
    best_iter <- as.integer(xgb.attr(fit, "best_iteration")) + 1L
    preds <- predict(fit, dva, iterationrange = c(1L, best_iter))
    colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
    actual <- as.matrix(train[va, .(Ch1,Ch2,Ch3,Ch4)])
    losses[k] <- mlogloss(actual, preds)
    cat("xgb fold", k, "loss:", losses[k], " best_iter:", best_iter, "\n")
  }
  losses
}
cv_xgb_base  <- run_xgb_cv(xgb_data_base)
cat("xgb BASE  - mean:", mean(cv_xgb_base), " (expect ~1.176378)\n")
cv_xgb_panel <- run_xgb_cv(xgb_data_panel)
cat("xgb PANEL - mean:", mean(cv_xgb_panel), "\n")
cat("Delta:", mean(cv_xgb_base) - mean(cv_xgb_panel), "\n\n")

# ==========================================================================
cat("=== SUMMARY ===\n")
cat("clogit: base", mean(cv_clogit_base), "-> panel", mean(cv_clogit_panel), "\n")
cat("ranger: base", mean(cv_ranger_base), "-> panel", mean(cv_ranger_panel), "\n")
cat("xgb:    base", mean(cv_xgb_base), "-> panel", mean(cv_xgb_panel), "\n")