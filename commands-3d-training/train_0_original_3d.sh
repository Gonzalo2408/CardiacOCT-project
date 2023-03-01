~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=low \
        gonzalorodriguez 10157 168 doduo1.umcn.nl/nnunet/sol \
        nnunet plan_train Task503_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-3d \
        --network 3d_fullres --trainer nnUNetTrainerV2SparseNormalSampling --ignore_data_integrity_check --fold all