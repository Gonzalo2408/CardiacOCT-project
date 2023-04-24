~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=high \
        gonzalorodriguez 10157 24 doduo2.umcn.nl/nnunet/sol \
        nnunet predict Task504_CardiacOCT \
        --results /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/results \
        --input /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task504_CardiacOCT/imagesTs \
        --output /mnt/netcache/diag/grodriguez/CardiacOCT/predicted_results_model4_2d --network 2d --disable_augmentation
