~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=high \
        gonzalorodriguez 10157 168 doduo2.umcn.nl/grodriguez/nnunet:latest \
        nnunet plan_train Task505_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-3d \
        --network 3d_fullres --trainer nnUNetTrainerV2Sparse --fold 0 --ignore_data_integrity_check