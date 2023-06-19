~/c-submit --require-cpus=1 --require-mem=30G --priority=low \
 gonzalorodriguez 10160 24 doduo2.umcn.nl/grodriguez/converter:model_pdf \
 --preds /mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/model8_preds \
 --pdf_name model8

 #--gpu-count=1 --require-gpu-mem=11g