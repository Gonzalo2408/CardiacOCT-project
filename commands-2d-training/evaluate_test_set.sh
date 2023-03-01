~/c-submit --require-cpus=9 --require-mem=28g --priority=high \
        gonzalorodriguez 10157 1 doduo1.umcn.nl/nnunet/sol \
        nnunet evaluate --ground_truth /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task503_CardiacOCT/labelsTs \
        --prediction /mnt/netcache/diag/grodriguez/CardiacOCT/predicted_results_model2_2d --labels 1-12