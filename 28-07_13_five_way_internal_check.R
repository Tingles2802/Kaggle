# 28-07_13_five_way_internal_check.R
#
# Purpose: internal-consistency check on the 5-way blend result from
# 28-07_12 (best: 1.154955 @ w_clogit=0.4, w_ranger=0.1, w_xgb=0.2, w_lc=0,
# w_ts=0.3). Confirms the improvement over Model 21 (1.156089) is actually
# attributable to Model 27's inclusion, not an artifact of re-searching the
# other four weights and landing on a different local optimum.
#
# PREREQS: run immediately after 28-07_12 in the same session -- reuses
# weights_grid, clogit_mat, ranger_mat, xgb_mat, lc_mat, actual_mat, mlogloss().

stopifnot(exists("weights_grid"), exists("clogit_mat"), exists("ranger_mat"),
          exists("xgb_mat"), exists("lc_mat"), exists("actual_mat"), exists("mlogloss"))

# Restrict the same grid to w_ts == 0 -- this should recover something close
# to Model 21's known weights (clogit15=0.6, ranger18=0.1, xgb20=0.3, lc13=0)
# and known CV (1.156089), as a sanity check that the OOF matrices used here
# are the same ones that originally produced Model 21.
sub <- weights_grid[w_ts == 0]
cat("w_ts==0 subgrid size:", nrow(sub), "combinations\n")

best4 <- list(loss = Inf, w = NULL)
for (i in seq_len(nrow(sub))) {
  w <- sub[i]
  blended <- w$w_clogit * clogit_mat + w$w_ranger * ranger_mat +
             w$w_xgb * xgb_mat + w$w_lc * lc_mat
  loss <- mlogloss(actual_mat, blended)
  if (loss < best4$loss) best4 <- list(loss = loss, w = w)
}

cat("\n4-way-only (w_ts=0) best log-loss:", best4$loss, "\n")
cat("Known Model 21 CV:                   1.156089\n")
print(best4$w)
cat("Known Model 21 weights: w_clogit=0.6, w_ranger=0.1, w_xgb=0.3, w_lc=0\n\n")

gap <- best4$loss - 1.156089
cat("Gap vs known Model 21:", gap, "\n")
if (abs(gap) < 0.0005 && best4$w$w_clogit == 0.6 && best4$w$w_ranger == 0.1 &&
    best4$w$w_xgb == 0.3 && best4$w$w_lc == 0) {
  cat("--> MATCHES closely. The OOF matrices used in 28-07_12 are consistent with\n")
  cat("    what produced Model 21. The 5-way result's 0.001134 improvement over\n")
  cat("    1.156089 is attributable to Model 27's inclusion, not a different local\n")
  cat("    optimum among the other four.\n")
} else {
  cat("--> DOES NOT closely match known Model 21. Something differs between the\n")
  cat("    oof_clogit_ranger_xgb.csv used here and whatever produced 1.156089/the\n")
  cat("    0.6/0.1/0.3/0 weights originally (different data version? different fold\n")
  cat("    seed used to generate that file? stale file?). Investigate before trusting\n")
  cat("    the 5-way blend result -- the 0.001134 gain may not be real.\n")
}
