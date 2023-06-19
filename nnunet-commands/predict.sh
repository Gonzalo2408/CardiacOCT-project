~/c-submit --require-cpus=4 --require-mem=20g --priority=high \
        gonzalorodriguez 10157 24 doduo2.umcn.nl/nnunet/sol:latest \
        nnunet predict Task512_CardiacOCT \
        --results /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/results \
        --input /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task512_CardiacOCT/imagesTs \
        --trainer nnUNetTrainer_V2_Loss_CEandDice_Weighted \
        --output /mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/model8_preds \
        --network 2d --disable_augmentation \
        --store_probability_maps

# --gpu-count=1 --require-gpu-mem=11g