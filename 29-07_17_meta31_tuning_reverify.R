# 29-07_17_meta31_tuning_reverify.R
#
# REQUIRES: oof_clogit_ranger_xgb.csv, oof_two_stage.csv, oof_glmnet.csv on disk.
# mlogloss() must be defined. Standalone — same build as 29-07_11/29-07_12.
#
# Purpose: fresh-seed (99) reverification of best tuning-grid config from
# 29-07_12 (depth=3, eta=0.03, nrounds=200, CV 1.151683 at seed 2024) vs
# current submitted config (depth=2, eta=0.05, nrounds=150, CV 1.152292).
# Checks whether the 0.0006 gain is real or within fresh-seed noise (~0.0005 seen before).

library(data.table); library(xgboost)

train <- fread("train.csv")
set.seed(99)
case_ids <- unique(train$Case)
case_fold <- data.table(Case = case_ids, fold = sample(rep(1:5, length.out = length(case_ids))))
train <- merge(train, case_fold, by = "Case")

oof_base <- fread("oof_clogit_ranger_xgb.csv")
oof_ts   <- fread("oof_two_stage.csv")
oof_gl   <- fread("oof_glmnet.csv")

oof_all <- merge(oof_base, oof_ts, by = c("Case", "Task"))
oof_all <- merge(oof_all, oof_gl, by = c("Case", "Task"))

fold_map <- unique(train[, .(Case, fold)])
oof_all  <- merge(oof_all, fold_map, by = "Case")
stopifnot(nrow(oof_all) == nrow(train))

feat_cols <- c("clogit15_Ch1","clogit15_Ch2","clogit15_Ch3","clogit15_Ch4",
               "ranger18_Ch1","ranger18_Ch2","ranger18_Ch3","ranger18_Ch4",
               "xgb20_Ch1","xgb20_Ch2","xgb20_Ch3","xgb20_Ch4",
               "ts_Ch1","ts_Ch2","ts_Ch3","ts_Ch4",
               "glmnet30_Ch1","glmnet30_Ch2","glmnet30_Ch3","glmnet30_Ch4")

y_class0   <- max.col(oof_all[, .(Ch1, Ch2, Ch3, Ch4)]) - 1
actual_mat <- as.matrix(oof_all[, .(Ch1, Ch2, Ch3, Ch4)])
folds      <- sort(unique(oof_all$fold))

configs <- list(
  tuned     = list(max_depth = 3, eta = 0.03, nrounds = 200),
  submitted = list(max_depth = 2, eta = 0.05, nrounds = 150)
)

for (nm in names(configs)) {
  cfg <- configs[[nm]]
  params <- list(objective = "multi:softprob", num_class = 4,
                  eta = cfg$eta, max_depth = cfg$max_depth,
                  subsample = 0.8, colsample_bytree = 0.8)
  meta_oof <- matrix(NA_real_, nrow(oof_all), 4)

  for (f in folds) {
    tr <- oof_all$fold != f
    te <- oof_all$fold == f
    dtr <- xgb.DMatrix(as.matrix(oof_all[tr, ..feat_cols]), label = y_class0[tr])
    dte <- xgb.DMatrix(as.matrix(oof_all[te, ..feat_cols]))
    fit  <- xgb.train(params, dtr, nrounds = cfg$nrounds)
    pred <- predict(fit, dte)
    meta_oof[te, ] <- if (!is.null(dim(pred))) pred else matrix(pred, ncol = 4, byrow = TRUE)
  }

  cv <- mlogloss(actual_mat, meta_oof)
  cat(sprintf("[seed 99] %-9s depth=%d eta=%.2f nrounds=%d -> CV %.6f\n",
              nm, cfg$max_depth, cfg$eta, cfg$nrounds, cv))
}

cat("\nseed 2024 reference: tuned 1.151683, submitted 1.152292\n")
