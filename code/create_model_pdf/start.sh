~/c-submit --require-cpus=1 --require-mem=30G --priority=low \
 gonzalorodriguez 10160 24 doduo2.umcn.nl/grodriguez/converter:model_pdf \
 --orig /mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task601_CardiacOCT/labelsTs \
 --preds /mnt/netcache/diag/grodriguez/CardiacOCT/preds_second_split/model_rgb_2d_last \
 --pdf_name model_rgb_last \

#--gpu-count=1 --require-gpu-mem=11g