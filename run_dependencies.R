# 1. Maintain your list of R files (in order)
script_list <- c(
  "27-07_1_pipeline_foundation.R",               # 1
  "27-07_2_clogit_baseline.R",                   # 2
  "27-07_3_clogit_interactions.R",               # 3
  "27-07_4_trees_setup.R",                       # 4
  "27-07_5_xgb onehot.R",                        # 5
  "27-07_6_alt4 check.R",                        # 6
  "27-07_7_clogit_asc4.R",                       # 7
  "27-07_8_interaction_screen.R",                # 8
  "27-07_9_clogit_promising_interactions.R",     # 9
  "27-07_10_clogit_simplified_segment.R",        # 10
  "27-07_12_latent_class_mnl.R",                 # 11  (Note: Original sequence skips 11)
  "27-07_13_demo_price_interactions_screen.R",   # 12
  "27-07_14_demo_price_interactions_isolate.R",  # 13
  "27-07_15_price_incomea_on_model15.R",         # 14
  "27-07_16_ranger_tuning.R",                    # 15
  "27-07_17_ranger_tuning_extended.R",           # 16
  "27-07_18_xgboost_tuning.R",                   # 17
  "27-07_19_xgboost_depth_extend.R",             # 18
  "28-07_1_oof_clogit_ranger_xgb.R",             # 19
  "28-07_2_oof_latent_class.R",                  # 20
  "28-07_3_ensemble_blend.R",                    # 21
  "28-07_4_submission_generator.R",              # 22
  "28-07_5_ensemble_reverify.R",                 # 23
  "28-07_6_partial_profile_check.R",             # 24
  "28-07_7_alt4_optout_demo.R",                  # 25
  "28-07_8_stacked_meta_learner.R",              # 26
  "28-07_9_attribute_demo_interactions.R",       # 27
  "28-07_10_relative_features_trees.R",          # 28
  "28-07_11_two_stage_optout.R"                  # 29
)

# 2. Select the indices of the files you want to run
# Example: To run files 1, 2, and the tuning script at 15
run_indices <- c(1, 2, 15)

# 3. Execute the selected files
for (idx in run_indices) {
  file_to_run <- script_list[idx]
  
  if (file.exists(file_to_run)) {
    cat(sprintf("\n[Running %d/%d]: %s\n", match(idx, run_indices), length(run_indices), file_to_run))
    sys.source(file_to_run, envir = globalenv())
  } else {
    warning(sprintf("File not found: %s", file_to_run))
  }
}

# ==============================================================================
# SCRIPT DEPENDENCY TREE (until 28-07_11)
# ==============================================================================
#
# 27-07_1_pipeline_foundation.R (Base Foundation)
# ├── 27-07_2_clogit_baseline.R
# │   └── 27-07_3_clogit_interactions.R
# ├── 27-07_4_trees_setup.R
# │   ├── 27-07_5_xgb_onehot.R
# │   │   ├── 27-07_18_xgboost_tuning.R
# │   │   └── 27-07_19_xgboost_depth_extend.R
# │   ├── 27-07_16_ranger_tuning.R
# │   ├── 27-07_17_ranger_tuning_extended.R
# │   └── 28-07_10_relative_features_trees.R
# ├── 27-07_6_alt4_check.R
# ├── 27-07_7_clogit_asc4.R
# ├── 27-07_8_interaction_screen.R
# │   └── 27-07_9_clogit_promising_interactions.R
# └── 27-07_10_clogit_simplified_segment.R
#     └── 27-07_13_demo_price_interactions_screen.R
#         ├── 27-07_14_demo_price_interactions_isolate.R
#         │   └── 27-07_15_price_incomea_on_model15.R
#         ├── 28-07_1_oof_clogit_ranger_xgb.R (Also requires 27-07_4 & 27-07_5)
#         ├── 28-07_5_ensemble_reverify.R     (Also requires 27-07_4 & 27-07_5)
#         ├── 28-07_7_alt4_optout_demo.R
#         │   └── 28-07_11_two_stage_optout.R
#         └── 28-07_9_attribute_demo_interactions.R
#
# [CSV Output Dependent Ensembles]
# 28-07_1 (OOF Clogit/Ranger/XGB) ──┐
#                                   ├──> 28-07_3_ensemble_blend.R
# 28-07_2 (OOF Latent Class) ───────┼──> 28-07_8_stacked_meta_learner.R
#                                   
# [Fully Standalone Scripts]
# - 28-07_2_oof_latent_class.R       (Self-contained; run in fresh R session)
# - 28-07_6_partial_profile_check.R  (Only reads raw train.csv)
#
# ------------------------------------------------------------------------------
# DEPENDENCY DETAILS & REQUIRED IN-MEMORY OBJECTS:
#
# [Track A: Clogit / Demographic Models]
# - 27-07_13_demo_price_interactions_screen.R
#   * Order: 27-07_1 -> 27-07_10
#   * Objects: train_long_ext, score_clogit_cv(), rhs_vars, actual_lookup
# - 27-07_14_demo_price_interactions_isolate.R
#   * Order: 27-07_1 -> 27-07_10 -> 27-07_13
#   * Objects: base_terms
# - 27-07_15_price_incomea_on_model15.R
#   * Order: 27-07_1 -> 27-07_10 -> 27-07_13 -> 27-07_14
# - 28-07_7_alt4_optout_demo.R
#   * Order: 27-07_1 -> 27-07_10 -> 27-07_13
# - 28-07_9_attribute_demo_interactions.R
#   * Order: 27-07_1 -> 27-07_10 -> 27-07_13
# - 28-07_11_two_stage_optout.R
#   * Order: 27-07_1 -> 27-07_10 -> 27-07_13 -> 28-07_7
#
# [Track B: Tree Models (Ranger & XGBoost)]
# - 27-07_16_ranger_tuning.R & 27-07_17_ranger_tuning_extended.R
#   * Order: 27-07_1 -> 27-07_4
#   * Objects: ranger_data, X_cols, mlogloss(), fold
# - 27-07_18_xgboost_tuning.R & 27-07_19_xgboost_depth_extend.R
#   * Order: 27-07_1 -> 27-07_4 (Steps 1-2) -> 27-07_5
#   * Objects: xgb_data_oh, numeric_cols, y_zero_indexed
# - 28-07_10_relative_features_trees.R
#   * Order: 27-07_1 -> 27-07_4
#
# [Ensemble Pipelines]
# - 28-07_1_oof_clogit_ranger_xgb.R
#   * Order: 27-07_1 -> 27-07_4 (Steps 1-2) -> 27-07_5 -> 27-07_10 -> 27-07_13
# - 28-07_5_ensemble_reverify.R
#   * Order: 27-07_1 -> 27-07_4 (Steps 1-2) -> 27-07_5 -> 27-07_10 -> 27-07_13
# ==============================================================================