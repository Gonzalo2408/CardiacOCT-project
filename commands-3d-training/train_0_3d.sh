~/c-submit --require-cpus=10 --require-mem=32g --gpu-count=1 --require-gpu-mem=12g --priority=high \
        gonzalorodriguez 10157 168 doduo1.umcn.nl/grodriguez/nnunet \
        nnunet plan_train Task505_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-3d \
        --network 3d --trainer nnUNetTrainerV2Sparse --fold 0