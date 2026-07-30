# 1. Maintain your list of R files
script_list <- c(
  "27-07_1_pipeline_foundation.R",               
  "27-07_2_clogit_baseline.R",                   
  "27-07_3_clogit_interactions.R",               
  "27-07_4_trees_setup.R",                       
  "27-07_5_xgb onehot.R",                        
  "27-07_6_alt4 check.R",                        
  "27-07_7_clogit_asc4.R",                       
  "27-07_8_interaction_screen.R",                
  "27-07_9_clogit_promising_interactions.R",     
  "27-07_10_clogit_simplified_segment.R",        
  "27-07_12_latent_class_mnl.R",                 
  "27-07_13_demo_price_interactions_screen.R",   
  "27-07_14_demo_price_interactions_isolate.R",  
  "27-07_15_price_incomea_on_model15.R",         
  "27-07_16_ranger_tuning.R",                    
  "27-07_17_ranger_tuning_extended.R",           
  "27-07_18_xgboost_tuning.R",                   
  "27-07_19_xgboost_depth_extend.R",             
  "28-07_1_oof_clogit_ranger_xgb.R",             
  "28-07_2_oof_latent_class.R",                  
  "28-07_3_ensemble_blend.R",                    
  "28-07_4_submission_generator.R",              
  "28-07_5_ensemble_reverify.R",                 
  "28-07_6_partial_profile_check.R",             
  "28-07_7_alt4_optout_demo.R",                  
  "28-07_8_stacked_meta_learner.R",              
  "28-07_9_attribute_demo_interactions.R",       
  "28-07_10_relative_features_trees.R",          
  "28-07_11_two_stage_optout.R",                  
  "28-07_12_oof_two_stage_and_blend.R",
  "28-07_13_five_way_internal_check.R",
  "29-07_1_reverify_model29.R",
  "29-07_2_submission_model29.R",
  "29-07_3_glmnet_multinomial.R",
  "29-07_4_oof_and_blend_glmnet.R",
  "29-07_5_gmnl_diagnostic.R",
  "29-07_7_gmnl_full_cv.R",
  "29-07_8_xgb_meta_learner.R",
  "29-07_9_xgb_meta_reverify.R",
  "29-07_10_submission_model31.R",
  "29-07_11_meta_5model_cv.R",
  "29-07_12_meta31_tuning.R",
  "29-07_13_rpar_diagnostic.R",
  "29-07_14_rpar_oof_predict_check.R",
  "29-07_15_rpar_fullscale_singlefold.R",
  "29-07_16_rpar_fullscale_score_patch.R",
  "29-07_17_meta31_tuning_reverify.R",
  "29-07_18_rpar_fold1_R200.R"
)

# 2. Select the files you want to run by name
run_files <- c(
  #"27-07_1_pipeline_foundation.R",
  #"27-07_4_trees_setup.R",
  "27-07_5_xgb onehot.R",
  "27-07_10_clogit_simplified_segment.R",
  "27-07_13_demo_price_interactions_screen.R",
  "28-07_5_ensemble_reverify.R",
  "28-07_7_alt4_optout_demo.R",
  "28-07_11_two_stage_optout.R",
  "28-07_1_oof_clogit_ranger_xgb.R",
  "28-07_2_oof_latent_class.R",
  "28-07_12_oof_two_stage_and_blend.R",
  "29-07_3_glmnet_multinomial.R",
  "29-07_4_oof_and_blend_glmnet.R",
  "29-07_2_submission_model29.R",
  "29-07_10_submission_model31.R",
  "29-07_20_submission_model32.R"
)

# 3. Execute the selected files
for (i in seq_along(run_files)) {
  file_to_run <- run_files[i]
  
  if (file.exists(file_to_run)) {
    cat(sprintf("\n[Running %d/%d]: %s\n", i, length(run_files), file_to_run))
    sys.source(file_to_run, envir = globalenv())
  } else {
    warning(sprintf("File not found: %s", file_to_run))
  }
}

# ==============================================================================
# SCRIPT DEPENDENCY TREE (until 29-07_18)
# ==============================================================================
#
# 27-07_1_pipeline_foundation.R (Base Foundation)
# ├── 27-07_2_clogit_baseline.R
# │   └── 27-07_3_clogit_interactions.R
# ├── 27-07_4_trees_setup.R (Also required by glmnet multinomial)
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
#         │   └── 29-07_1_reverify_model29.R
#         ├── 28-07_7_alt4_optout_demo.R
#         │   └── 28-07_11_two_stage_optout.R
#         │       ├── 28-07_12_oof_two_stage_and_blend.R
#         │       │   └── 28-07_13_five_way_internal_check.R
#         │       └── 29-07_1_reverify_model29.R
#         │           └── 29-07_2_submission_model29.R
#         └── 28-07_9_attribute_demo_interactions.R
#
# [CSV Output Dependent Ensembles]
# 28-07_1 (OOF Clogit/Ranger/XGB) ──┐
#                                   ├──> 28-07_3_ensemble_blend.R
# 28-07_2 (OOF Latent Class) ───────┼──> 28-07_8_stacked_meta_learner.R
#                                   └──> 28-07_12_oof_two_stage_and_blend.R ──> 28-07_13_five_way_internal_check.R
#                                                                            ├──> 29-07_8_xgb_meta_learner.R
#                                                                            ├──> 29-07_9_xgb_meta_reverify.R
#                                                                            ├──> 29-07_10_submission_model31.R
#                                                                            ├──> 29-07_11_meta_5model_cv.R
#                                                                            ├──> 29-07_12_meta31_tuning.R
#                                                                            └──> 29-07_17_meta31_tuning_reverify.R
#                                   
# [Fully Standalone / Alternative Tracks]
# - 28-07_2_oof_latent_class.R       (Self-contained; run in fresh R session)
# - 28-07_6_partial_profile_check.R  (Only reads raw train.csv)
# - 29-07_3_glmnet_multinomial.R     (Requires 27-07_1 and 27-07_4 steps 1-3)
#   └── 29-07_4_oof_and_blend_glmnet.R
# - 29-07_5_gmnl_diagnostic.R        (Requires 27-07_1, 27-07_10, 27-07_12/13)
#   └── 29-07_7_gmnl_full_cv.R         (Requires 27-07_1 and 27-07_10 clean state)
# - 29-07_13_rpar_diagnostic.R       (Standalone diagnostic script for mlogit::rpar)
# - 29-07_14_rpar_oof_predict_check.R(Standalone OOF prediction diagnostic for mlogit::rpar)
# - 29-07_15_rpar_fullscale_singlefold.R (Standalone single-fold full-scale diagnostic for mlogit::rpar)
# - 29-07_16_rpar_fullscale_score_patch.R (In-console patch script for mlogit::rpar single-fold scoring)
# - 29-07_18_rpar_fold1_R200.R      (Standalone higher-draw single-fold diagnostic for mlogit::rpar)
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
# - 28-07_12_oof_two_stage_and_blend.R
#   * Order: 27-07_1 -> 27-07_10 -> 27-07_13 -> 28-07_7 -> 28-07_11
# - 28-07_13_five_way_internal_check.R
#   * Order: Run immediately after 28-07_12
# - 29-07_1_reverify_model29.R
#   * Order: 27-07_1, 27-07_10, 27-07_12/13, 28-07_7, 28-07_11, then 28-07_5
# - 29-07_2_submission_model29.R
#   * Order: 27-07_1, 27-07_4, 27-07_5, 27-07_10, 27-07_13, 28-07_7, 28-07_11
# - 29-07_5_gmnl_diagnostic.R & 29-07_7_gmnl_full_cv.R
#   * Order: 27-07_1, 27-07_10 (requires train_long, fold, rhs_vars, mloss, train)
# - 29-07_8 to 29-07_12 scripts & 29-07_17_meta31_tuning_reverify.R
#   * Order: Require disk files `oof_clogit_ranger_xgb.csv`, `oof_two_stage.csv`, `oof_glmnet.csv`, and `train.csv`.
# - 29-07_13 to 29-07_18 scripts
#   * Order: Standalone diagnostics, patches, and high-draw tests for `mlogit::rpar`.
#
# [Ensemble Pipelines]
# - 28-07_1_oof_clogit_ranger_xgb.R
#   * Order: 27-07_1 -> 27-07_4 (Steps 1-2) -> 27-07_5 -> 27-07_10 -> 27-07_13
# - 28-07_5_ensemble_reverify.R
#   * Order: 27-07_1 -> 27-07_4 (Steps 1-2) -> 27-07_5 -> 27-07_10 -> 27-07_13
# ==============================================================================