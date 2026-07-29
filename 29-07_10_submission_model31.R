# 29-07_10_submission_model31.R
#
# REQUIRES: run 29-07_2_submission_model29.R FIRST, in the same session, unmodified
# up through its Section 4 (two-stage). This script reuses, without recomputation:
#   test, test_long, rhs_vars, identified_vars, cat_attr_vars, attr_vars, price_cols,
#   ranger_data, X_cols, cat_feature_cols, xgb_data_oh, numeric_cols, y_zero_indexed,
#   onehot_dt, build_onehot(), agea_center, agea_scale,
#   preds_clogit, preds_ranger, preds_xgb, preds_ts (test-set base predictions),
#   mlogloss().
# Also requires oof_clogit_ranger_xgb.csv, oof_two_stage.csv, oof_glmnet.csv to
# exist on disk (already generated in prior sessions). oof_latent_class.csv is
# NOT used — lc13 dropped, see Section 1.
#
# Purpose: fit the 1 base model NOT already produced by 29-07_2 (glmnet ridge) on
# FULL train + test, then fit the Model 31 xgboost meta-learner (5 base models,
# NOT 6 — latent-class/lc13 DROPPED, see Section 1 note) on the FULL training OOF
# stack (no held-out fold — final refit) and score test. Writes submission_29-07_1.csv.
#
# DEVIATION FROM LOGGED MODEL 31 (CV 1.152278/1.152751): that CV run used 6 base
# models (24 feat_cols) including lc13. This submission uses 5 (20 feat_cols) —
# gmnl's full-data fit is broken on this environment (see Section 1). CV has NOT
# been re-run on the 5-model feature set before this submission — expected to be
# very close (lc13 was zero-weighted in every linear ensemble) but NOT reverified.
# Log this deviation explicitly; do not treat submitted score as validating the
# original 1.152278 CV number.

library(data.table); library(survival); library(glmnet); library(xgboost)

set.seed(2024)

# =====================================================================
# 1. Latent-class MNL (Model 13) — DROPPED from Model 31.
# Full-data gmnl fit fails (scale-dependent maxLik "eval(ei, envir)" error;
# confirmed NOT a version/API issue - 400-row slice fits cleanly, full
# ~21.5k rows breaks). lc13 had zero weight in every linear ensemble tried
# (Models 21/28/29). Not worth further debugging given the deadline.
# If `test` was previously renamed to test_data while diagnosing this, restore it:
if (exists("test_data")) { test <- test_data; rm(test_data) }

# =====================================================================
# 2. glmnet ridge multinomial (Model 30) — refit on FULL train, predict on test
# =====================================================================

raw_X_full <- ranger_data[, !c("y", "fold"), with = FALSE]
is_constant_full <- vapply(raw_X_full, function(col) length(unique(col)) <= 1, logical(1))
if (any(is_constant_full)) raw_X_full <- raw_X_full[, !names(raw_X_full)[is_constant_full], with = FALSE]
X_full_train <- model.matrix(~ . - 1, data = raw_X_full)
y_full_train <- ranger_data$y
stopifnot(identical(levels(y_full_train), c("1","2","3","4")))

fit_glmnet_full <- cv.glmnet(X_full_train, y_full_train, family = "multinomial",
                             type.multinomial = "grouped", alpha = 0, nfolds = 5, standardize = TRUE)

test_X_glmnet <- test_X[, !is_constant_full, with = FALSE]  # test_X built + releveled in 29-07_2 Section 2
X_test_glmnet <- model.matrix(~ . - 1, data = test_X_glmnet)
stopifnot(identical(colnames(X_test_glmnet), colnames(X_full_train)))

preds_gl <- predict(fit_glmnet_full, newx = X_test_glmnet, s = "lambda.min", type = "response")[, , 1]
colnames(preds_gl) <- c("glmnet30_Ch1","glmnet30_Ch2","glmnet30_Ch3","glmnet30_Ch4")
preds_glmnet <- data.table(Case = test$Case, Task = test$Task, No = test$No, preds_gl)

# =====================================================================
# 3. Meta-learner (Model 31) — fit xgboost on FULL training OOF stack, score test
# =====================================================================

train_oof <- fread("train.csv")  # fresh, unmodified copy for fold-id + Ch1-4 (matches 29-07_8 convention)
set.seed(2024)
case_ids <- unique(train_oof$Case)
case_fold <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train_oof <- merge(train_oof, case_fold, by = "Case")

oof_base <- fread("oof_clogit_ranger_xgb.csv")
oof_ts_f <- fread("oof_two_stage.csv")
oof_gl_f <- fread("oof_glmnet.csv")

oof_all <- merge(oof_base, oof_ts_f, by = c("Case","Task"))
oof_all <- merge(oof_all, oof_gl_f, by = c("Case","Task"))
stopifnot(nrow(oof_all) == nrow(train_oof))

feat_cols <- c("clogit15_Ch1","clogit15_Ch2","clogit15_Ch3","clogit15_Ch4",
               "ranger18_Ch1","ranger18_Ch2","ranger18_Ch3","ranger18_Ch4",
               "xgb20_Ch1","xgb20_Ch2","xgb20_Ch3","xgb20_Ch4",
               "ts_Ch1","ts_Ch2","ts_Ch3","ts_Ch4",
               "glmnet30_Ch1","glmnet30_Ch2","glmnet30_Ch3","glmnet30_Ch4")

y_class0_full <- max.col(oof_all[, .(Ch1, Ch2, Ch3, Ch4)]) - 1
params <- list(objective = "multi:softprob", num_class = 4,
               eta = 0.05, max_depth = 2, subsample = 0.8, colsample_bytree = 0.8)
nrounds <- 150  # untuned/shallow by design, unchanged from 29-07_8/9

d_meta_train <- xgb.DMatrix(as.matrix(oof_all[, ..feat_cols]), label = y_class0_full)
fit_meta_full <- xgb.train(params, d_meta_train, nrounds = nrounds)  # final refit, no held-out fold

# ---- Assemble test-set feature matrix in the SAME feat_cols order ----
preds_all6 <- merge(preds_clogit[, .(Case, Task, No, clogit15_Ch1, clogit15_Ch2, clogit15_Ch3, clogit15_Ch4)],
                    preds_ranger, by = c("Case","Task","No"), sort = FALSE)
preds_all6 <- merge(preds_all6, preds_xgb, by = c("Case","Task","No"), sort = FALSE)
preds_all6 <- merge(preds_all6, preds_ts, by = c("Case","Task","No"), sort = FALSE)
preds_all6 <- merge(preds_all6, preds_glmnet, by = c("Case","Task","No"), sort = FALSE)
stopifnot(nrow(preds_all6) == nrow(test))

d_meta_test <- xgb.DMatrix(as.matrix(preds_all6[, ..feat_cols]))
meta_pred <- predict(fit_meta_full, d_meta_test)

cat("predict() class:", class(meta_pred), " dim:", paste(dim(meta_pred), collapse = "x"), "\n")
if (is.null(dim(meta_pred))) meta_pred <- matrix(meta_pred, ncol = 4, byrow = TRUE)  # gotcha #19 guard
colnames(meta_pred) <- c("Ch1","Ch2","Ch3","Ch4")

# =====================================================================
# 4. Format + write
# =====================================================================

cat("Row-sum range pre-normalize (should already be ~1):", range(rowSums(meta_pred)), "\n")
meta_pred <- meta_pred / rowSums(meta_pred)

stopifnot(!any(is.na(meta_pred)))
stopifnot(all(meta_pred >= -1e-8 & meta_pred <= 1 + 1e-8))

submission <- data.table(No = preds_all6$No, meta_pred)
setorder(submission, No)

stopifnot(nrow(submission) == 4997)
stopifnot(identical(names(submission), c("No","Ch1","Ch2","Ch3","Ch4")))
stopifnot(all(abs(rowSums(submission[, .(Ch1,Ch2,Ch3,Ch4)]) - 1) < 1e-8))

fwrite(submission, "submission_29-07_1.csv")
cat("\nSaved submission_29-07_1.csv (", nrow(submission), "rows )\n")
print(head(submission))