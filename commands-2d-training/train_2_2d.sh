~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=10g --priority=low \
        gonzalorodriguez 10157 168 doduo2.umcn.nl/nnunet/sol:latest \
        nnunet plan_train Task511_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d \
        --network 2d \
        --dont_plan_3d --fold 2