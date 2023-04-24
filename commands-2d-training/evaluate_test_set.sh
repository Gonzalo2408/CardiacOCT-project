~/c-submit --require-cpus=9 --require-mem=28g --priority=low \
        gonzalorodriguez 10157 1 doduo2.umcn.nl/nnunet/sol \
        nnunet evaluate --ground_truth /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task504_CardiacOCT/labelsTs \
        --prediction /mnt/netcache/diag/grodriguez/CardiacOCT/predicted_results_model4_2d --labels 1-12