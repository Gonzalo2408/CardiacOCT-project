~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=high \
        gonzalorodriguez 10157 168 doduo1.umcn.nl/grodriguez/nnunet:latest \
        plan_train Task502_CardiacOCT \
        /mnt/netcache/diag/grodriguez/data \
        --network 3d_fullres --trainer nnUNetTrainerV2Sparse --ignore_data_integrity_check --fold 0