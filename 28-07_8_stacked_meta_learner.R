# Requires: 28-07_1 (oof_clogit_ranger_xgb.csv), 28-07_2 (oof_latent_class.csv).
# train.csv loaded fresh here; fold assignment self-generated (grouped by Case, seed 2024)
# — does not need to match original modeling folds exactly, only needs to be a valid grouped split.
library(data.table)
library(glmnet)

train <- fread("train.csv")

set.seed(2024)
case_ids <- unique(train$Case)
case_fold <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train <- merge(train, case_fold, by = "Case")

oof_base <- fread("oof_clogit_ranger_xgb.csv")
oof_lc   <- fread("oof_latent_class.csv")
oof_all  <- merge(oof_base, oof_lc, by = c("Case", "Task"))

fold_map <- unique(train[, .(Case, fold)])
oof_all  <- merge(oof_all, fold_map, by = "Case")

feat_cols <- c("clogit15_Ch1","clogit15_Ch2","clogit15_Ch3","clogit15_Ch4",
               "ranger18_Ch1","ranger18_Ch2","ranger18_Ch3","ranger18_Ch4",
               "xgb20_Ch1","xgb20_Ch2","xgb20_Ch3","xgb20_Ch4",
               "lc13_Ch1","lc13_Ch2","lc13_Ch3","lc13_Ch4")

y_class <- max.col(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])  # true class 1..4
meta_oof <- matrix(NA_real_, nrow(oof_all), 4)

for (f in sort(unique(oof_all$fold))) {
  tr <- oof_all$fold != f
  te <- oof_all$fold == f
  
  x_tr <- as.matrix(oof_all[tr, ..feat_cols])
  x_te <- as.matrix(oof_all[te, ..feat_cols])
  y_tr <- factor(y_class[tr], levels = 1:4)
  
  cvfit <- cv.glmnet(x_tr, y_tr, family = "multinomial", alpha = 0, type.measure = "deviance")
  pred  <- predict(cvfit, newx = x_te, s = "lambda.min", type = "response")
  meta_oof[te, ] <- pred[,,1]
}

actual_mat <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
meta_cv <- mlogloss(actual_mat, meta_oof)
meta_cv  # compare directly to Model 21's 1.156089