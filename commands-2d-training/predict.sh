~/c-submit --require-cpus=9 --require-mem=32g --gpu-count=1 --require-gpu-mem=11g --priority=high --node=dlc-lugia\
        gonzalorodriguez 10157 24 doduo2.umcn.nl/grodriguez/nnunet:latest \
        nnunet predict Task508_CardiacOCT \
        --results /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/results \
        --input /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task508_CardiacOCT/imagesTs \
        --output /mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/pred_trial --network 2d --disable_augmentation \
        --trainer nnUNetTrainer_V2_Loss_CEandDice_Weighted --store_probability_maps

#  \