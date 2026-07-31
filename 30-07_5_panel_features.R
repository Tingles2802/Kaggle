# Prerequisites: NONE. Self-contained — fresh fread of train.csv/test.csv.
# Purpose: Case-level (respondent) design-based features, computed from OFFERED
# attributes/prices only (never from Ch/choice) -> leak-free for train AND test,
# since all 19 tasks' profiles are visible upfront regardless of response.
# Distinct from Models 25/26 (PriceRank/PriceGap = cross-ALT within a task, RULED OUT).
# This is cross-TASK within a respondent (Case).
# Output: writes case_panel_features_train.csv / case_panel_features_test.csv,
# row-aligned to train/test by Case+Task. Merge downstream via guarded merge.

library(data.table)

build_panel_features <- function(dt) {
  # Price4 excluded: Alt4 is structural opt-out, Price4 fixed at 0 every task (not a real
  # offer) -> including it deflates the baseline and makes PriceDev4 redundant/constant.
  price_cols <- c("Price1","Price2","Price3")
  
  # Case-level price baseline (mean/sd across 57 real offered price cells: 19 tasks x 3 alts)
  case_stats <- dt[, .(
    case_mean_price = mean(unlist(.SD)),
    case_sd_price    = sd(unlist(.SD))
  ), by = Case, .SDcols = price_cols]
  
  dt <- merge(dt, case_stats, by = "Case", all.x = TRUE)
  
  # Per-alt price deviation from this respondent's own baseline (alts 1-3 only)
  for (k in 1:3) {
    dt[, paste0("PriceDev", k) := get(paste0("Price", k)) - case_mean_price]
  }
  
  # Task-position features (case-task-level constants; interact with Price for clogit,
  # usable raw for ranger/xgboost)
  dt[, TaskFrac    := Task / 19]
  dt[, IsFirstTask := as.integer(Task == 1)]
  
  dt[, .(Case, Task, No,
         PriceDev1, PriceDev2, PriceDev3,
         case_mean_price, case_sd_price, TaskFrac, IsFirstTask)]
}

train <- fread("train.csv")
test  <- fread("test.csv")

panel_train <- build_panel_features(train)
panel_test  <- build_panel_features(test)

stopifnot(nrow(panel_train) == nrow(train), nrow(panel_test) == nrow(test))

fwrite(panel_train, "case_panel_features_train.csv")
fwrite(panel_test,  "case_panel_features_test.csv")

cat("Train panel features:", nrow(panel_train), "rows\n")
cat("Test panel features:", nrow(panel_test), "rows\n")
print(head(panel_train))