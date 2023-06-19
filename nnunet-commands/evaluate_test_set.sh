~/c-submit --require-cpus=4 --require-mem=28g --priority=low \
        gonzalorodriguez 10157 1 doduo2.umcn.nl/nnunet/sol:latest \
        nnunet evaluate --ground_truth /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task512_CardiacOCT/labelsTs \
        --prediction /mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/model8_preds --labels 1-12