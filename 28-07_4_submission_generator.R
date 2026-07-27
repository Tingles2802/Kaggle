# 28-07_4_submission_generator.R
# REQUIRES (run first, in order): 27-07_1_pipeline_foundation.R,
#   27-07_4_trees_setup.R, 27-07_5_xgb_onehot.R,
#   27-07_10_clogit_simplified_segment.R, 27-07_13_demo_price_interactions_screen.R
#   — same chain as 28-07_1. Needs (from that chain, still in scope, don't
#   rm(list=ls())): train, train_long, train_long_ext, rhs_vars, attr_vars,
#   cat_attr_vars, X_cols, demo_cols, cat_feature_cols, numeric_cols,
#   price_cols, ranger_data, xgb_data_oh, onehot_dt, build_onehot(),
#   y_zero_indexed, mlogloss().
#
# Purpose: refit clogit15 / ranger18 / xgb20 on FULL train (no fold holdout),
# predict on test.csv, blend with Model 21's weights (clogit .6/ranger .1/
# xgb .3/lc 0 — latent-class excluded, see Model Registry), write
# submission_28-07_1.csv in sample_submission.csv's exact format.
#
# CRITICAL: every scaling parameter / factor level set used below is taken
# FROM TRAIN and reapplied to test — never recomputed on test. Recomputing
# on test is a silent leakage/mismatch bug, not a valid shortcut.

library(survival)
library(ranger)
library(xgboost)
library(data.table)

test <- fread("test.csv")
stopifnot(nrow(test) == 4997)

# =====================================================================
# 1. clogit (Model 15) — refit on full train, predict on test
# =====================================================================

model15_formula <- as.formula(paste0(
  "chosen ~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender + strata(chid)"
))
model15_score_formula <- as.formula(paste0(
  "~ ", paste(rhs_vars, collapse = " + "),
  " + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender"
))

fit_clogit_full <- clogit(model15_formula, data = train_long_ext, method = "exact")
stopifnot(!any(is.na(coef(fit_clogit_full))))  # no aliased coefs on full data

build_long_test <- function(dt, alt) {
  cols <- paste0(attr_vars, alt)
  out <- dt[, ..cols]
  setnames(out, cols, attr_vars)
  out[, `:=`(Case = dt$Case, Task = dt$Task, No = dt$No,
             chid = paste0(dt$Case, "_", dt$Task), alt = alt)]
  out
}
test_long <- rbindlist(lapply(1:4, build_long_test, dt = test))
setorder(test_long, chid, alt)

# categorical attrs -> factor using TRAIN's levels (not test's own)
for (v in cat_attr_vars) {
  tr_lv <- levels(train_long[[v]])
  stopifnot(all(as.character(unique(test_long[[v]])) %in% tr_lv))
  test_long[, (v) := factor(get(v), levels = tr_lv)]
}

# Price scaling: reconstruct TRAIN's original center/scale (train_long$Price
# was overwritten in-place by scale() in 27-07_1, so recover it from the
# still-raw wide Price1-4 columns, which is the exact population scale()
# was applied to).
raw_price_train <- unlist(train[, ..price_cols], use.names = FALSE)
price_center <- mean(raw_price_train)
price_scale  <- sd(raw_price_train)
test_long[, Price := as.numeric(Price)]
test_long[, Price := (Price - price_center) / price_scale]

# demo merge (only what Model 15 needs: segment, agea, Urbind, gender)
demo_cols_clogit <- c("Case", "segment", "agea", "Urbind", "gender")
stopifnot(all(demo_cols_clogit %in% names(test)))
demo_lookup_test <- unique(test[, ..demo_cols_clogit])
test_long <- merge(test_long, demo_lookup_test, by = "Case", sort = FALSE)

# agea_z: train_long$agea was never overwritten (only agea_z is derived),
# so its center/scale can be read directly off train_long$agea.
agea_center <- mean(train_long$agea)
agea_scale  <- sd(train_long$agea)
test_long[, agea_z := (agea - agea_center) / agea_scale]

test_long[, is_luxury_segment := as.numeric(grepl("Luxury", segment, ignore.case = TRUE))]
test_long[, gender := factor(gender, levels = levels(train_long_ext$gender))]
stopifnot(!any(is.na(test_long$gender)))  # no unseen gender level

mm <- model.matrix(model15_score_formula, data = test_long)
mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
keep_coef <- names(coef(fit_clogit_full))[!is.na(coef(fit_clogit_full))]
mm <- mm[, keep_coef, drop = FALSE]
test_long[, score := as.numeric(mm %*% coef(fit_clogit_full)[keep_coef])]
test_long[, prob := exp(score - max(score)) / sum(exp(score - max(score))), by = chid]

setorder(test_long, chid, alt)
preds_c <- dcast(test_long, chid ~ alt, value.var = "prob")
setnames(preds_c, as.character(1:4), c("clogit15_Ch1","clogit15_Ch2","clogit15_Ch3","clogit15_Ch4"))
ct_key <- unique(test_long[, .(chid, Case, Task, No)])
setkey(ct_key, chid)
preds_clogit <- merge(ct_key, preds_c, by = "chid", sort = FALSE)

# =====================================================================
# 2. ranger (Model 18) — refit on full train, predict on test
# =====================================================================

test_X <- copy(test[, ..X_cols])
for (v in cat_feature_cols) {
  tr_lv <- levels(ranger_data[[v]])
  stopifnot(all(as.character(unique(test_X[[v]])) %in% tr_lv))
  test_X[, (v) := factor(get(v), levels = tr_lv)]
}

fit_ranger_full <- ranger(y ~ ., data = ranger_data[, !c("fold"), with = FALSE],
                          probability = TRUE, num.trees = 500,
                          mtry = 40, min.node.size = 60, seed = 2024)
preds_r <- predict(fit_ranger_full, data = test_X)$predictions
colnames(preds_r) <- c("ranger18_Ch1","ranger18_Ch2","ranger18_Ch3","ranger18_Ch4")
preds_ranger <- data.table(Case = test$Case, Task = test$Task, No = test$No, preds_r)

# =====================================================================
# 3. xgboost (Model 20) — refit on full train, predict on test
# =====================================================================

onehot_dt_test <- copy(test[, ..cat_feature_cols])
for (v in cat_feature_cols) {
  tr_lv <- levels(onehot_dt[[v]])
  stopifnot(all(as.character(unique(onehot_dt_test[[v]])) %in% tr_lv))
  onehot_dt_test[, (v) := factor(get(v), levels = tr_lv)]
}
onehot_mat_test <- build_onehot(onehot_dt_test, cat_feature_cols)
xgb_data_oh_test <- cbind(onehot_mat_test, as.matrix(test[, ..numeric_cols]))
stopifnot(identical(colnames(xgb_data_oh_test), colnames(xgb_data_oh)))  # hard column-alignment check

# No held-out fold for early stopping on the full-data fit — use Model 20's
# CV-averaged best iteration (180.4 -> 180) as a fixed nrounds instead.
dtrain_full <- xgb.DMatrix(data = xgb_data_oh, label = y_zero_indexed)
fit_xgb_full <- xgb.train(
  params = list(objective = "multi:softprob", num_class = 4,
                eval_metric = "mlogloss", eta = 0.1, max_depth = 3),
  data = dtrain_full, nrounds = 180, verbose = 0
)
preds_x <- predict(fit_xgb_full, xgb.DMatrix(data = xgb_data_oh_test))
colnames(preds_x) <- c("xgb20_Ch1","xgb20_Ch2","xgb20_Ch3","xgb20_Ch4")
preds_xgb <- data.table(Case = test$Case, Task = test$Task, No = test$No, preds_x)

# =====================================================================
# 4. Blend (Model 21 weights: clogit 0.6 / ranger 0.1 / xgb 0.3 / lc 0)
# =====================================================================

preds_all <- merge(preds_clogit[, .(Case, Task, No, clogit15_Ch1, clogit15_Ch2, clogit15_Ch3, clogit15_Ch4)],
                   preds_ranger, by = c("Case","Task","No"), sort = FALSE)
preds_all <- merge(preds_all, preds_xgb, by = c("Case","Task","No"), sort = FALSE)
stopifnot(nrow(preds_all) == nrow(test))

blend <- 0.6 * as.matrix(preds_all[, .(clogit15_Ch1,clogit15_Ch2,clogit15_Ch3,clogit15_Ch4)]) +
  0.1 * as.matrix(preds_all[, .(ranger18_Ch1,ranger18_Ch2,ranger18_Ch3,ranger18_Ch4)]) +
  0.3 * as.matrix(preds_all[, .(xgb20_Ch1,xgb20_Ch2,xgb20_Ch3,xgb20_Ch4)])
colnames(blend) <- c("Ch1","Ch2","Ch3","Ch4")

cat("Row-sum range pre-normalize (should already be ~1):", range(rowSums(blend)), "\n")
blend <- blend / rowSums(blend)  # defensive renormalize regardless

stopifnot(!any(is.na(blend)))
stopifnot(all(blend >= -1e-8 & blend <= 1 + 1e-8))

# =====================================================================
# 5. Format + write — match sample_submission.csv exactly: columns
#    No, Ch1, Ch2, Ch3, Ch4; 4997 rows; probabilities sum to 1/row.
# =====================================================================

submission <- data.table(No = preds_all$No, blend)
setorder(submission, No)

stopifnot(nrow(submission) == 4997)
stopifnot(identical(names(submission), c("No","Ch1","Ch2","Ch3","Ch4")))
stopifnot(all(abs(rowSums(submission[, .(Ch1,Ch2,Ch3,Ch4)]) - 1) < 1e-8))

fwrite(submission, "submission_27-07_2.csv")
cat("\nSaved submission_28-07_1.csv (", nrow(submission), "rows )\n")
print(head(submission))