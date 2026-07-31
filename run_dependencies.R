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
  "29-07_18_rpar_fold1_R200.R",
  "29-07_19_nnet_full.R",
  "29-07_20_submission_model32.R",
  "30-07_1_knn_oof.R",
  "30-07_3_meta_ext.R",
  "30-07_4_meta_ext_tune.R",
  "30-07_4_rpar_diagnostic.R",
  "30-07_5_panel_features.R",
  "30-07_6_panel_features_cv_test.R",
  "30-07_7_meta_panel_test.R",
  "30-07_8_meta_panel_tune.R",
  "30-07_9_model38_freshseed.R",
  "30-07_10_model38_submission.R",
  "31-07_1_lag_features_ab.R",
  "31-07_2_lag_recursive_cv.R",
  "31-07_3_panel_base_retrofit.R",
  "31-07_4_bagged_ranger_test.R",
  "31-07_5_segment_oof_slice.R",
  "31-07_6_segment_price_interaction.R",
  "31-07_6_weighted_xgb_ab.R",
  "31-07_7_final_verification.R"
)

# 2. Safety Backup Pre-Check (STEP 0)
if (file.exists("submission_30-07_2.csv") && !file.exists("submission_30-07_2_ORIGINAL_BACKUP.csv")) {
  file.copy("submission_30-07_2.csv", "submission_30-07_2_ORIGINAL_BACKUP.csv")
  cat("[STEP 0] Created safety backup: submission_30-07_2_ORIGINAL_BACKUP.csv\n")
}

# Master execution chain for 31-07_7_final_verification.R
files_to_run <- c(
  "30-07_7_meta_panel_test.R",
  "30-07_8_meta_panel_tune.R",
  "30-07_10_model38_submission.R",
  "31-07_7_final_verification.R"
)

# 4. Select the batch to execute (Default: Verification Chain)
run_files <- files_to_run

# 5. Execute the selected pipeline
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
# SCRIPT DEPENDENCY TREE (until 31-07_7_final_verification)
# ==============================================================================
#
# 27-07_1_pipeline_foundation.R (Base Foundation)
# ├── 27-07_2_clogit_baseline.R
# │   └── 27-07_3_clogit_interactions.R
# ├── 27-07_4_trees_setup.R (Required by glmnet multinomial, KNN OOF, 30-07_6, 31-07_4 & 31-07_5 chain)
# │   ├── 27-07_5_xgb onehot.R
# │   │   ├── 27-07_18_xgboost_tuning.R
# │   │   └── 27-07_19_xgboost_depth_extend.R
# │   ├── 27-07_16_ranger_tuning.R
# │   ├── 27-07_17_ranger_tuning_extended.R
# │   ├── 28-07_10_relative_features_trees.R
# │   ├── 30-07_6_panel_features_cv_test.R (Requires 27-07_1, 27-07_4 + case_panel_features_train.csv from 30-07_5)
# │   └── 31-07_4_bagged_ranger_test.R     (Requires 27-07_1 and 27-07_4 loaded in workspace)
# ├── 27-07_6_alt4 check.R
# ├── 27-07_7_clogit_asc4.R
# ├── 27-07_8_interaction_screen.R
# │   └── 27-07_9_clogit_promising_interactions.R
# └── 27-07_10_clogit_simplified_segment.R
#     ├── 27-07_13_demo_price_interactions_screen.R
#     │   ├── 27-07_14_demo_price_interactions_isolate.R
#     │   │   └── 27-07_15_price_incomea_on_model15.R
#     │   ├── 28-07_1_oof_clogit_ranger_xgb.R (Requires 27-07_4 & 27-07_5)
#     │   ├── 28-07_5_ensemble_reverify.R     (Requires 27-07_4 & 27-07_5)
#     │   │   └── 29-07_1_reverify_model29.R
#     │   ├── 28-07_7_alt4_optout_demo.R
#     │   │   └── 28-07_11_two_stage_optout.R
#     │   │       ├── 28-07_12_oof_two_stage_and_blend.R
#     │   │       │   └── 28-07_13_five_way_internal_check.R
#     │   │       └── 29-07_1_reverify_model29.R
#     │   │           └── 29-07_2_submission_model29.R
#     │   └── 28-07_9_attribute_demo_interactions.R
#     └── 31-07_6_segment_price_interaction.R (Requires 27-07_1, 27-07_10, and score_clogit_cv from 27-07_13)
#
# [CSV Output / In-Memory Dependent Ensembles]
# 28-07_1 (OOF Clogit/Ranger/XGB) ──┐
#                                   ├──> 28-07_3_ensemble_blend.R
# 28-07_2 (OOF Latent Class) ───────┼──> 28-07_8_stacked_meta_learner.R
#                                   └──> 28-07_12_oof_two_stage_and_blend.R ──> 28-07_13_five_way_internal_check.R
#                                                                            ├──> 29-07_8_xgb_meta_learner.R
#                                                                            ├──> 29-07_9_xgb_meta_reverify.R
#                                                                            ├──> 29-07_10_submission_model31.R
#                                                                            │    └──> 29-07_20_submission_model32.R
#                                                                            │         └──> 30-07_4_meta_ext_tune.R (Requires 29-07_20 chain + 30-07_1 + 30-07_3)
#                                                                            │              └──> 30-07_7_meta_panel_test.R (Requires 30-07_3, 30-07_4 + 30-07_5 on disk)
#                                                                            │                   └──> 30-07_8_meta_panel_tune.R (Requires 30-07_7 run in memory)
#                                                                            │                        └──> 30-07_9_model38_freshseed.R (Requires 30-07_7/8 run in memory)
#                                                                            │                             ├──> 30-07_10_model38_submission.R (Requires 29-07_10 test preds_all6, 30-07_7 in memory + 30-07_5 on disk)
#                                                                            │                             │    └──> 31-07_7_final_verification.R (Requires Model 38 canonical chain & submission_30-07_2.csv)
#                                                                            │                             └──> 31-07_5_segment_oof_slice.R (Requires 27-07_1, 27-07_4, 30-07_7/8/9 in memory)
#                                                                            ├──> 29-07_11_meta_5model_cv.R
#                                                                            ├──> 29-07_12_meta31_tuning.R
#                                                                            ├──> 29-07_17_meta31_tuning_reverify.R
#                                                                            └──> 30-07_3_meta_ext.R (Requires chain oof_all, fold_vec, feat_cols + nnet & knn OOF)
#                                                                                 └──> 30-07_4_meta_ext_tune.R
#                                                                                      └──> 30-07_7_meta_panel_test.R
#                                                                                           └──> 30-07_8_meta_panel_tune.R
#                                                                                                └──> 30-07_9_model38_freshseed.R
#                                                                                                     ├──> 30-07_10_model38_submission.R ──> 31-07_7_final_verification.R
#                                                                                                     └──> 31-07_5_segment_oof_slice.R
#                                   
# [Fully Standalone / Alternative Tracks]
# - 28-07_2_oof_latent_class.R       (Self-contained; run in fresh R session)
# - 28-07_6_partial_profile_check.R  (Only reads raw train.csv)
# - 29-07_3_glmnet_multinomial.R     (Requires 27-07_1 and 27-07_4 steps 1-3)
#   └── 29-07_4_oof_and_blend_glmnet.R
# - 29-07_13 to 29-07_18 scripts     (Standalone diagnostics, patches, and high-draw tests for `mlogit::rpar`)
# - 29-07_19_nnet_full.R             (Standalone full-feature multinomial logit script using `nnet::multinom`)
# - 30-07_1_knn_oof.R                (Requires 27-07_1 and 27-07_4 for pipeline/tree foundation matrices)
# - 30-07_4_rpar_diagnostic.R        (Standalone diagnostic script; reads raw `train.csv` directly)
# - 30-07_5_panel_features.R         (Standalone feature generation script; reads raw `train.csv`/`test.csv` directly)
# - 31-07_1_lag_features_ab.R        (Standalone A/B diagnostic; reads raw `train.csv` directly)
# - 31-07_2_lag_recursive_cv.R       (Standalone recursive CV script; reads raw `train.csv` directly)
# - 31-07_3_panel_base_retrofit.R    (Requires `train.csv` and `case_panel_features_train.csv` on disk)
# - 31-07_4_bagged_ranger_test.R     (Requires 27-07_1 and 27-07_4 loaded in workspace)
# - 31-07_5_segment_oof_slice.R      (Requires 27-07_1, 27-07_4, and Model 38 environment loaded)
# - 31-07_6_segment_price_interaction.R (Requires 27-07_1, 27-07_10, and score_clogit_cv from 27-07_13)
# - 31-07_6_weighted_xgb_ab.R        (Standalone A/B diagnostic; reads raw `train.csv` directly)
# - 31-07_7_final_verification.R     (Requires Model 38 canonical chain execution and fresh submission file to diff against backup)
# ==============================================================================