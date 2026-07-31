# Prerequisites: 27-07_1, 27-07_4, ... through 29-07_10 (needs preds_all6 in memory,
# test-side 5-model table w/ feat_cols cols + No). Also needs dtrain_ext2/feat_cols_ext2
# from 30-07_7 in memory (same session), and case_panel_features_test.csv on disk
# (from 30-07_5). Self-contained for the nnet/knn parts (fresh fread, matches
# 29-07_19/30-07_1's own pattern) - does NOT depend on unconfirmed intermediate objects.

library(data.table); library(nnet); library(class); library(xgboost)

train_raw <- fread("train.csv")
test_raw  <- fread("test.csv")

attr_names <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")
attr_cols  <- as.vector(outer(attr_names, 1:4, paste0))
price_cols <- paste0("Price", 1:4)
demo_cols  <- c("segment","year","miles","night","ppark","gender","age","educ","region","Urb","income")

# ---- nnet: full refit on all train, predict on test ----
for (col in demo_cols) {
  lv <- union(unique(train_raw[[col]]), unique(test_raw[[col]]))  # shared levels, avoid new-level errors
  train_raw[[col]] <- factor(train_raw[[col]], levels = lv)
  test_raw[[col]]  <- factor(test_raw[[col]],  levels = lv)
}
train_raw[, y := factor(max.col(as.matrix(.SD)), levels = 1:4), .SDcols = c("Ch1","Ch2","Ch3","Ch4")]

feat_cols_nnet <- c(attr_cols, price_cols, demo_cols)
f_nnet <- as.formula(paste("y ~", paste(feat_cols_nnet, collapse = " + ")))
set.seed(2024)
fit_nnet_full <- multinom(f_nnet, data = train_raw, maxit = 300, trace = FALSE)
pred_nnet_test <- predict(fit_nnet_full, newdata = test_raw, type = "probs")
stopifnot(ncol(pred_nnet_test) == 4)
nnet_test <- data.table(No = test_raw$No,
                          nnet_Ch1 = pred_nnet_test[,1], nnet_Ch2 = pred_nnet_test[,2],
                          nnet_Ch3 = pred_nnet_test[,3], nnet_Ch4 = pred_nnet_test[,4])

# ---- k-NN: combined dummy-encode train+test attr cols for consistent columns, k=50 ----
attr_only <- rbind(train_raw[, ..attr_cols], test_raw[, ..attr_cols])
attr_only <- attr_only[, lapply(.SD, factor)]
is_constant <- sapply(attr_only, function(col) length(unique(col)) <= 1)  # Gotcha #20
attr_only <- attr_only[, .SD, .SDcols = !is_constant]

X_all <- model.matrix(~ ., data = attr_only)[, -1]
n_tr <- nrow(train_raw)
X_tr <- scale(X_all[1:n_tr, ])
X_te <- scale(X_all[(n_tr+1):nrow(X_all), ],
              center = attr(X_tr, "scaled:center"), scale = attr(X_tr, "scaled:scale"))

y_knn <- as.integer(train_raw$y)
set.seed(2024)
pred_knn <- knn(X_tr, X_te, cl = as.factor(y_knn), k = 50, prob = TRUE)
won <- attr(pred_knn, "prob")
cls <- as.integer(as.character(pred_knn))
m <- matrix((1 - won) / 3, length(pred_knn), 4)
for (i in seq_along(pred_knn)) m[i, cls[i]] <- won[i]
knn_test <- data.table(No = test_raw$No, knn_Ch1 = m[,1], knn_Ch2 = m[,2],
                         knn_Ch3 = m[,3], knn_Ch4 = m[,4])

# ---- panel features (already built) ----
panel_test <- fread("case_panel_features_test.csv")

# ---- merge everything onto preds_all6, all keyed by No ----
stopifnot("preds_all6" %in% ls())  # must exist from 29-07_10 chain
d_test_ext2 <- merge(preds_all6[, c("No", feat_cols), with = FALSE], nnet_test, by = "No")
d_test_ext2 <- merge(d_test_ext2, knn_test, by = "No")
d_test_ext2 <- merge(d_test_ext2, panel_test[, .(No, PriceDev1, PriceDev2, PriceDev3,
                                                   case_mean_price, case_sd_price,
                                                   TaskFrac, IsFirstTask)], by = "No")
stopifnot(nrow(d_test_ext2) == 4997)

X_test_final <- as.matrix(d_test_ext2[, ..feat_cols_ext2])

# ---- refit Model 38 on ALL training rows, predict ----
params_38 <- list(objective = "multi:softprob", num_class = 4,
                    eval_metric = "mlogloss", eta = 0.02, max_depth = 3)
dtrain_full <- xgb.DMatrix(as.matrix(d_train_ext2[, ..feat_cols_ext2]), label = y_ext)
fit_meta38_full <- xgb.train(params_38, dtrain_full, nrounds = 286)

preds_38 <- predict(fit_meta38_full, X_test_final, reshape = TRUE)
colnames(preds_38) <- c("Ch1","Ch2","Ch3","Ch4")

submission_38 <- data.table(No = d_test_ext2$No, preds_38)
stopifnot(nrow(submission_38) == 4997, all.equal(rowSums(preds_38), rep(1,4997), tolerance = 1e-6))

fwrite(submission_38, "submission_30-07_2.csv")
cat("Written submission_30-07_2.csv,", nrow(submission_38), "rows\n")
print(head(submission_38))
