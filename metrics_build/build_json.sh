#!/bin/bash

#results_folder model 1 2D: "Z:/grodriguez/CardiacOCT/predicted_results_model1_2d"
#new_json_file model 1 2D: "Z:/grodriguez/CardiacOCT/metrics_build/new_pullback_metrics_model1_test.json"

#results_folder model 1 2D cv: "Z:/grodriguez/CardiacOCT/data-2d/results/nnUNet/2d/Task501_CardiacOCT/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_postprocessed"
#new_json_file model 1 2D cv: "Z:/grodriguez/CardiacOCT/metrics_build/new_pullback_metrics_model1_cv.json"

python build_results.py \
    --results_folder "Z:/grodriguez/CardiacOCT/predicted_results_model1_2d" \
    --mode "pullback" \
    --new_json_file "Z:/grodriguez/CardiacOCT/metrics_build/new_pullback_recall_model1_test.json"