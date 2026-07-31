# 31-07_3_panel_base_retrofit_v2.R
# STANDALONE - fresh fread of train.csv + panel CSVs. Saves OOF predictions.
# Run in CLEAN session.
#
# Modifications from v1:
#   - Saves OOF probability matrices for clogit, ranger, xgboost (panel versions)
#   - Files: oof_clogit_panel.csv, oof_ranger_panel.csv, oof_xgb_panel.csv
#   - Format: Case, Task, Ch1, Ch2, Ch3, Ch4 (same as existing OOF files)

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

train[, PriceDev4 := 0]

# ==========================================================================
# PART A: clogit (panel version) - save OOF probs
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

for (v in attr_vars) train_long[, (v) := as.factor(get(v))]
train_long[, Price := as.numeric(scale(Price))]

rhs_vars <- c(attr_vars, "Price")

# Panel formula = base + PriceDev + Price:TaskFrac + Price:case_sd_price
panel_formula <- as.formula(paste0("chosen ~ ", paste(rhs_vars, collapse = " + "),
                                   " + PriceDev + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
                                   " + Price:TaskFrac + Price:case_sd_price + strata(chid)"))
panel_score   <- as.formula(paste0("~ ", paste(rhs_vars, collapse = " + "),
                                   " + PriceDev + Price:agea_z + Price:is_luxury_segment + Price:Urbind + Price:gender",
                                   " + Price:TaskFrac + Price:case_sd_price"))

score_clogit_cv_with_oof <- function(fit_formula, score_formula, data_long, actual_full) {
  losses <- numeric(5)
  oof_list <- list()
  
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
    
    oof_list[[i]] <- pw[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]
  }
  list(losses = losses, oof = rbindlist(oof_list))
}

actual_lookup <- train[, .(Case, Task, Ch1, Ch2, Ch3, Ch4)]

res_clogit <- score_clogit_cv_with_oof(panel_formula, panel_score, train_long, actual_lookup)
cat("clogit PANEL - mean:", mean(res_clogit$losses), " sd:", sd(res_clogit$losses), "\n\n")

fwrite(res_clogit$oof, "oof_clogit_panel.csv")
cat("Saved: oof_clogit_panel.csv\n\n")

# ==========================================================================
# PART B: ranger (panel version) - save OOF probs
# ==========================================================================
cat_feature_cols <- unlist(lapply(1:4, function(a) paste0(attr_vars, a)))
price_cols <- paste0("Price", 1:4)
pricedev_cols <- paste0("PriceDev", 1:4)
panel_extra <- c("case_mean_price","case_sd_price","TaskFrac","IsFirstTask")

ranger_panel_cols <- c(cat_feature_cols, price_cols, demo_cols, pricedev_cols, panel_extra)

build_ranger_data <- function(cols) {
  rd <- copy(train[, ..cols])
  for (v in cat_feature_cols) rd[, (v) := as.factor(get(v))]
  rd[, y := factor((train$Ch1==1)*1 + (train$Ch2==1)*2 + (train$Ch3==1)*3 + (train$Ch4==1)*4, levels = 1:4)]
  rd[, fold := train$fold]
  rd
}

ranger_data_panel <- build_ranger_data(ranger_panel_cols)

run_ranger_cv_with_oof <- function(rd) {
  losses <- numeric(5)
  oof_list <- list()
  
  for (k in 1:5) {
    tr <- rd[fold != k]; va <- rd[fold == k]
    fit <- ranger(y ~ ., data = tr[, !c("fold"), with = FALSE], probability = TRUE,
                  num.trees = 500, mtry = 40, min.node.size = 60, seed = 2024)
    preds <- predict(fit, data = va)$predictions
    colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
    actual <- as.matrix(train[fold == k, .(Ch1,Ch2,Ch3,Ch4)])
    losses[k] <- mlogloss(actual, preds)
    cat("ranger fold", k, "loss:", losses[k], "\n")
    
    oof_list[[k]] <- data.table(
      Case = va$Case,
      Task = va$Task,
      Ch1 = preds[,1],
      Ch2 = preds[,2],
      Ch3 = preds[,3],
      Ch4 = preds[,4]
    )
  }
  list(losses = losses, oof = rbindlist(oof_list))
}

res_ranger <- run_ranger_cv_with_oof(ranger_data_panel)
cat("ranger PANEL - mean:", mean(res_ranger$losses), "\n\n")

fwrite(res_ranger$oof, "oof_ranger_panel.csv")
cat("Saved: oof_ranger_panel.csv\n\n")

# ==========================================================================
# PART C: xgboost (panel version) - save OOF probs
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

onehot_mat <- build_onehot(train, cat_feature_cols)
numeric_panel <- as.matrix(train[, c(price_cols, demo_cols, pricedev_cols, panel_extra), with = FALSE])
xgb_data_panel <- cbind(onehot_mat, numeric_panel)
y_idx <- (train$Ch1==1)*0 + (train$Ch2==1)*1 + (train$Ch3==1)*2 + (train$Ch4==1)*3

run_xgb_cv_with_oof <- function(X) {
  losses <- numeric(5)
  oof_list <- list()
  
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
    
    oof_list[[k]] <- data.table(
      Case = train$Case[va],
      Task = train$Task[va],
      Ch1 = preds[,1],
      Ch2 = preds[,2],
      Ch3 = preds[,3],
      Ch4 = preds[,4]
    )
  }
  list(losses = losses, oof = rbindlist(oof_list))
}

res_xgb <- run_xgb_cv_with_oof(xgb_data_panel)
cat("xgb PANEL - mean:", mean(res_xgb$losses), "\n\n")

fwrite(res_xgb$oof, "oof_xgb_panel.csv")
cat("Saved: oof_xgb_panel.csv\n\n")

# ==========================================================================
cat("=== SUMMARY ===\n")
cat("clogit panel CV:", mean(res_clogit$losses), "\n")
cat("ranger panel CV:", mean(res_ranger$losses), "\n")
cat("xgb panel CV:   ", mean(res_xgb$losses), "\n")
cat("\nOOF files written:\n")
cat("  oof_clogit_panel.csv\n")
cat("  oof_ranger_panel.csv\n")
cat("  oof_xgb_panel.csv\n")