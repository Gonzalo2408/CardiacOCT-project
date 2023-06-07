~/c-submit --require-cpus=9 --require-mem=28g --priority=high \
        gonzalorodriguez 10157 1 doduo2.umcn.nl/nnunet/sol \
        nnunet evaluate --ground_truth /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task510_CardiacOCT/labelsTs \
        --prediction /mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/predicted_results_only_lipid --labels 1