~/c-submit --require-cpus=4 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=high --node=dlc-ditto\
        gonzalorodriguez 10157 1 doduo2.umcn.nl/nnunet/sol:latest \
        nnunet find_best_configuration Task604_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d \
        --networks 2d