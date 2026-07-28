# 28-07_11_two_stage_optout.R
# PREREQS (source first, only if not already loaded this session):
#   27-07_1  -> train, fold
#   27-07_10 -> rhs_vars, agea_z, is_luxury_segment, Urbind
#   27-07_12/27-07_13 -> train_long_ext (long format, outcome col = chosen)
#   28-07_7  -> is_male (added to train_long_ext)
# VERIFY before running: names(train_long_ext) has cols "alt" (1:4) and "chid".
# If train was ever touched by a failed/partial run of this script before,
# rm(train) and re-source 27-07_1 first — do not resume on a stale train.
# NOTE: train_long_ext may already carry "fold" from an earlier session's script —
# that's fine; the guards below check/merge each needed column independently.

library(data.table); library(survival)

# ---- defensive check: catch corrupted merges from any prior unguarded run ----
if (any(c("is_male.x","is_male.y","fold.x","fold.y","Ch4.x","Ch4.y") %in% names(train))) {
  stop("train has stale duplicate columns from a prior unguarded merge — rm(train) and re-source 27-07_1 before continuing.")
}
if (any(c("fold.x","fold.y","Ch4.x","Ch4.y","is_male.x","is_male.y") %in% names(train_long_ext))) {
  stop("train_long_ext has stale duplicate columns from a prior unguarded merge — rebuild via 27-07_12/13 + 28-07_7 before continuing.")
}

# ---- idempotent setup merges: each column checked/merged independently, safe to re-run ----
if (!"is_male" %in% names(train)) {
  is_male_lookup <- unique(train_long_ext[, .(Case, Task, is_male)])
  train <- merge(train, is_male_lookup, by = c("Case","Task"))
}
if (!"is_luxury_segment" %in% names(train)) {
  is_luxury_lookup <- unique(train_long_ext[, .(Case, Task, is_luxury_segment)])
  train <- merge(train, is_luxury_lookup, by = c("Case","Task"))
}
if (!"agea_z" %in% names(train)) {
  agea_z_lookup <- unique(train_long_ext[, .(Case, Task, agea_z)])
  train <- merge(train, agea_z_lookup, by = c("Case","Task"))
}
if (!"fold" %in% names(train_long_ext)) {
  fold_lookup <- unique(train[, .(Case, Task, fold)])
  train_long_ext <- merge(train_long_ext, fold_lookup, by = c("Case","Task"))
}
if (!"Ch4" %in% names(train_long_ext)) {
  Ch4_lookup <- unique(train[, .(Case, Task, Ch4)])
  train_long_ext <- merge(train_long_ext, Ch4_lookup, by = c("Case","Task"))
}

# ---- Stage A: opt-out (Ch4) binary model, Task-level ----
train[, minPrice123    := pmin(Price1, Price2, Price3)]
train[, meanPrice123   := (Price1 + Price2 + Price3) / 3]
train[, spreadPrice123 := pmax(Price1, Price2, Price3) - minPrice123]

stageA_formula <- Ch4 ~ minPrice123 + meanPrice123 + spreadPrice123 +
  is_male + is_luxury_segment + agea_z + Urbind + milesa + nighta + incomea

# ---- Stage B: conditional logit, Alts 1-3 only, reuses Model 15's rhs_vars spec ----
stageB_rhs <- c(rhs_vars, "Price:agea_z", "Price:is_luxury_segment", "Price:Urbind", "Price:gender")
stageB_formula <- as.formula(paste("chosen ~", paste(stageB_rhs, collapse = " + "), "+ strata(chid)"))
stageB_score_formula <- as.formula(paste("~", paste(stageB_rhs, collapse = " + ")))

stageB_pool <- train_long_ext[alt %in% 1:3]

# ---- CV loop (reuses existing case-grouped `fold`) ----
cv_two_stage <- function() {
  losses <- numeric(5)
  for (k in 1:5) {
    fitA <- glm(stageA_formula, data = train[fold != k], family = binomial())
    pA   <- predict(fitA, newdata = train[fold == k], type = "response")  # P(Ch4=1)
    
    fitB <- clogit(stageB_formula, data = stageB_pool[fold != k & Ch4 == 0])
    teB  <- stageB_pool[fold == k]
    mm <- model.matrix(stageB_score_formula, data = teB)[, -1, drop = FALSE]
    
    bB <- coef(fitB)
    bB[is.na(bB)] <- 0          # NEW: aliased/rank-deficient coefs -> 0 contribution, matches predict()'s internal handling
    teB[, linpred := as.vector(mm %*% bB)]
    teB[, expu := exp(linpred)]
    teB[, pB := expu / sum(expu), by = chid]
    
    predB_wide <- dcast(teB, Case + Task ~ alt, value.var = "pB")
    setnames(predB_wide, c("1","2","3"), c("p1","p2","p3"))
    
    test_ids <- train[fold == k, .(Case, Task, Ch4)]
    test_ids[, pOptOut := pA]
    combined <- merge(test_ids, predB_wide, by = c("Case","Task"))
    combined[, `:=`(
      predCh1 = (1 - pOptOut) * p1, predCh2 = (1 - pOptOut) * p2,
      predCh3 = (1 - pOptOut) * p3, predCh4 = pOptOut
    )]
    
    actual <- actual_lookup[combined, on = c("Case","Task")]
    losses[k] <- mlogloss(as.matrix(actual[, .(Ch1,Ch2,Ch3,Ch4)]),
                          as.matrix(combined[, .(predCh1,predCh2,predCh3,predCh4)]))
  }
  losses
}

fold_losses <- cv_two_stage()
mean(fold_losses); sd(fold_losses)