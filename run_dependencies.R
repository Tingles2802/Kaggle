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