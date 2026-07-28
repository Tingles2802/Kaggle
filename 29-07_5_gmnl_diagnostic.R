# 29-07_5_gmnl_diagnostic.R
# Prereqs: 27-07_1, 27-07_10 (rhs_vars), 27-07_12/13 (train_long_ext, chid)
# DIAGNOSTIC ONLY (Section 1.2) - small slice, inspect object before any full run.
# ASSUMPTION TO VERIFY: alt-index column assumed named "alt" - check train_long_ext's
# actual column name and fix chid.var/alt.var below if different.

library(gmnl)
library(mlogit)

small_cases <- unique(train_long_ext$Case)[1:200]
dat_small <- train_long_ext[Case %in% small_cases]

dat_small_ml <- mlogit.data(dat_small, choice = "chosen", shape = "long",
                            chid.var = "chid", alt.var = "alt", id.var = "Case")

f <- mFormula(as.formula(paste("chosen ~", paste(rhs_vars, collapse = " + "), "| 0")))

fit_diag <- gmnl(f, data = dat_small_ml, model = "mixl",
                 ranp = c(Price = "n"), R = 50, panel = TRUE,
                 print.level = 1)

# Paste back all of this before proceeding further:
packageVersion("gmnl")
class(fit_diag)
names(fit_diag)
summary(fit_diag)