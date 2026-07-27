# REQUIRES: run 27-07_1 then 27-07_4 first (builds ranger_data, X_cols, fold, mlogloss()).
# Ignore 27-07_4's trailing fold_loss_Bp error (Gotcha #16) - harmless, everything needed exists before it.
# Extends grid past Model 17's edge-optimum (mtry=23, min.node.size=20 -> CV 1.195617).

library(ranger)
library(data.table)

mtry_grid <- c(23, 30, 40)
node_grid <- c(20, 40, 60)
grid <- expand.grid(mtry = mtry_grid, min.node.size = node_grid)

results <- data.table(mtry = integer(), min.node.size = integer(), cv_loss = double(), cv_sd = double())

for (i in seq_len(nrow(grid))) {
  m <- grid$mtry[i]; nsz <- grid$min.node.size[i]
  fold_losses <- numeric(5)
  
  for (f in 1:5) {
    tr <- ranger_data[fold != f]
    va <- ranger_data[fold == f]
    
    # drop only 'fold' - keep y and all X_cols, matching 27-07_4's pattern
    fit <- ranger(y ~ ., data = tr[, !c("fold"), with = FALSE], probability = TRUE,
                  num.trees = 500, mtry = m, min.node.size = nsz, seed = 2024)
    
    preds <- predict(fit, data = va)$predictions  # columns already ordered "1","2","3","4"
    colnames(preds) <- c("Ch1","Ch2","Ch3","Ch4")
    
    actual <- as.matrix(train[fold == f, .(Ch1, Ch2, Ch3, Ch4)])
    fold_losses[f] <- mlogloss(actual, preds)
  }
  
  results <- rbind(results, data.table(mtry = m, min.node.size = nsz,
                                       cv_loss = mean(fold_losses), cv_sd = sd(fold_losses)))
  cat(sprintf("mtry=%d, min.node.size=%d -> CV %.6f (sd %.6f)\n", m, nsz, mean(fold_losses), sd(fold_losses)))
}

print(results[order(cv_loss)])