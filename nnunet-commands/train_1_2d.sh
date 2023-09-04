~/c-submit --require-cpus=8 --require-mem=28g --gpu-count=1 --require-gpu-mem=10g --priority=low --node=dlc-lugia\
        gonzalorodriguez 10157 168 doduo2.umcn.nl/nnunet/sol:latest \
        nnunet plan_train Task604_CardiacOCT \
        /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d \
        --network 2d \
        --dont_plan_3d --fold 1

#--trainer nnUNetTrainer_V2_Loss_CEandDice_Weighted \
#--trainer_kwargs "{\"ignore_label\":-100, \"weight_dc\":3, \"weight_ce\":1, \"class_weights\":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}" \