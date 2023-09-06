~/c-submit --require-cpus=4 --require-mem=28g --priority=high --node=dlc-ditto\
        gonzalorodriguez 10157 1 doduo2.umcn.nl/nnunet/sol:latest \
        nnunet evaluate --ground_truth /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task604_CardiacOCT/labelsTs \
        --prediction /mnt/netcache/diag/grodriguez/CardiacOCT/preds_second_split/model_pseudo3d_3_preds --labels 1-12