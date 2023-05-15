~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=low \
        gonzalorodriguez 10157 24 doduo2.umcn.nl/nnunet/sol \
        nnunet predict Task501_CardiacOCT \
        --results /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/results \
        --input /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task504_CardiacOCT/imagesTs \
        --output /mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/new_pullback_pred_model1 --network 2d --disable_augmentation \
        --store_probability_maps
