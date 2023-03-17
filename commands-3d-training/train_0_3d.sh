~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=high \
        gonzalorodriguez 10157 168 doduo1.umcn.nl/grodriguez/nnunet \
        nnunet plan_train Task504_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-3d \
        --network 3d_fullres --trainer nnUNetTrainerV2Sparse --fold all --ignore_data_integrity_check