# Standalone diagnostic - only needs train.csv. No deps on prior session objects.
library(data.table)
train <- fread("train.csv")

attr_cols <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")

# 1. value ranges per attribute per alt - look for a 0/NA-like "not shown" code
for (a in attr_cols) {
  cols <- paste0(a, 1:4)
  cat(a, ": ", paste(range(unlist(train[, ..cols])), collapse="-"), 
      " | unique vals: ", paste(sort(unique(unlist(train[, ..cols]))), collapse=","), "\n")
}

# 2. does any attribute have a value that appears far more often than others (candidate "inactive" code)?
for (a in attr_cols) {
  cols <- paste0(a, 1:4)
  print(table(unlist(train[, ..cols])))
}

# 3. Price - check if it's ever 0/NA (partial profile sometimes omits price too)
cat("Price range:", range(unlist(train[, .(Price1,Price2,Price3,Price4)])), "\n")

# Do zeros co-occur across attributes within the same alt-slot? (i.e. is the whole slot inactive?)
attr_cols <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")

for (alt in 1:4) {
  cols <- paste0(attr_cols, alt)
  all_zero <- rowSums(train[, ..cols] == 0) == length(attr_cols)
  any_zero <- rowSums(train[, ..cols] == 0) > 0
  cat("Alt", alt, "- rows where ALL 19 attrs = 0:", sum(all_zero),
      "| rows where SOME (but not all) = 0:", sum(any_zero & !all_zero), "\n")
}

# Price=0 frequency, and does it align with all-zero attribute slots?
for (alt in 1:4) {
  pcol <- paste0("Price", alt)
  acols <- paste0(attr_cols, alt)
  all_zero <- rowSums(train[, ..acols] == 0) == length(attr_cols)
  cat("Alt", alt, "- Price==0 count:", sum(train[[pcol]] == 0),
      "| Price==0 AND all attrs==0:", sum(train[[pcol]] == 0 & all_zero), "\n")
}

# Was a fully-zero alt-slot ever the chosen one? (should be 0/near-0 if these are placeholders)
chcols <- c("Ch1","Ch2","Ch3","Ch4")
for (alt in 1:4) {
  acols <- paste0(attr_cols, alt)
  all_zero <- rowSums(train[, ..acols] == 0) == length(attr_cols)
  cat("Alt", alt, "- chosen when all-zero:", sum(train[[chcols[alt]]] == 1 & all_zero),
      "out of", sum(all_zero), "all-zero rows\n")
}

# How many active (non-fully-zero) alternatives per row - is it always 4, or does it vary (2/3/4)?
active_count <- rowSums(sapply(1:4, function(alt) {
  acols <- paste0(attr_cols, alt)
  rowSums(train[, ..acols] == 0) < length(attr_cols)
}))
print(table(active_count))