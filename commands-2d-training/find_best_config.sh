~/c-submit --require-cpus=9 --require-mem=28g --priority=high \
        gonzalorodriguez 10157 1 doduo2.umcn.nl/nnunet/sol \
        nnunet find_best_configuration Task510_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d \
        --networks 2d 