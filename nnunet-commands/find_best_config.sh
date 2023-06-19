~/c-submit --require-cpus=4 --require-mem=28g --priority=low \
        gonzalorodriguez 10157 1 doduo2.umcn.nl/nnunet/sol \
        nnunet find_best_configuration Task512_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d \
        --trainer nnUNetTrainer_V2_Loss_CEandDice_Weighted \
        --networks 2d