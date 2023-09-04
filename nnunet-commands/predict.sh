~/c-submit --require-cpus=4 --require-mem=20g --priority=high --gpu-count=1 --require-gpu-mem=10g \
        gonzalorodriguez 10157 24 doduo2.umcn.nl/grodriguez/nnunet:latest \
        nnunet predict Task601_CardiacOCT \
        --results /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/results \
        --input /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/imgs_to_feature \
        --output /mnt/netcache/diag/grodriguez/CardiacOCT/feature_maps/radb_0089_601_v2 \
        --network 2d --disable_augmentation \
        --store_probability_maps
        #--checkpoint model_last \

#--gpu-count=1 --require-gpu-mem=11g