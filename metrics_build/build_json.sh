#!/bin/bash

#results_folder model 1 2D: 
#new_json_file model 1 2D: "Z:/grodriguez/CardiacOCT/metrics_build/new_pullback_metrics_model1_test.json"
#"Z:/grodriguez/CardiacOCT/data-2d/results/nnUNet/2d/Task506_CardiacOCT/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_postprocessed"

#results_folder model 1 2D cv: "Z:/grodriguez/CardiacOCT/data-2d/results/nnUNet/2d/Task501_CardiacOCT/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_postprocessed"
#new_json_file model 1 2D cv: "Z:/grodriguez/CardiacOCT/metrics_build/new_pullback_metrics_model1_cv.json"

python build_results.py \
    --mode "frame" \
    --new_json_file "Z:/grodriguez/CardiacOCT/metrics_build/frame_lipid_model_test_dice.json" \
    --folder predicted_results_only_lipid