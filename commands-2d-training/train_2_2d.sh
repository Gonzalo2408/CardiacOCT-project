~/c-submit --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g --priority=low \
        gonzalorodriguez 10157 168 doduo2.umcn.nl/nnunet/sol:latest \
        nnunet plan_train Task507_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d \
        --network 2d --trainer nnUNetTrainer_V2_Loss_CEandDice_Weighted \
        --trainer_kwargs "{\"ignore_label\":-100, \"class_weights\":[0.109, 0.695, 1.41, 1.322, 2.536, 10.889, 4.698, 5.802, 19.474, 739.368, 1553.895, 15430.272, 133.963], \"weight_dc\":3.0,\"weight_ce\":0.8}" --dont_plan_3d --fold 2