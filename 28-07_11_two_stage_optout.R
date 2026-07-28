# 28-07_11_two_stage_optout.R
# PREREQS: 27-07_1, 27-07_10, 27-07_13, 28-07_7 sourced first (train, fold, rhs_vars,
# agea_z, is_luxury_segment, Urbind, is_male, train_long_ext all must exist).
# VERIFY: train_long_ext's alt-id col is named "alt" (1:4) and occasion id is "chid" —
# run names(train_long_ext) first; rename below if different.
library(data.table); library(survival)

# ---- Stage A: opt-out (Ch4) binary model, Task-level ----
train[, minPrice123    := pmin(Price1, Price2, Price3)]
train[, meanPrice123   := (Price1 + Price2 + Price3) / 3]
train[, spreadPrice123 := pmax(Price1, Price2, Price3) - minPrice123]

stageA_formula <- Ch4 ~ minPrice123 + meanPrice123 + spreadPrice123 +
  is_male + is_luxury_segment + agea_z + Urbind + milesa + nighta + incomea

# ---- Stage B: conditional logit, Alts 1-3 only, reuses Model 15's rhs_vars spec ----
stageB_rhs <- c(rhs_vars, "Price:agea_z", "Price:is_luxury_segment", "Price:Urbind", "Price:gender")
stageB_formula <- as.formula(paste("chosen ~", paste(stageB_rhs, collapse = " + "), "+ strata(chid)"))

# merge fold/Ch4 onto long data once; pool = all alt1-3 rows (used for both fit-subset and predict-all)
train_long_ext <- merge(train_long_ext, train[, .(Case, Task, fold, Ch4)], by = c("Case","Task"))
stageB_pool <- train_long_ext[alt %in% 1:3]

# ---- CV loop (reuses existing case-grouped `fold`) ----
cv_two_stage <- function() {
  losses <- numeric(5)
  for (k in 1:5) {
    fitA <- glm(stageA_formula, data = train[fold != k], family = binomial())
    pA   <- predict(fitA, newdata = train[fold == k], type = "response")  # P(Ch4=1)
    
    fitB <- clogit(stageB_formula, data = stageB_pool[fold != k & Ch4 == 0])
    teB  <- stageB_pool[fold == k]
    teB[, linpred := predict(fitB, newdata = teB, type = "lp")]
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