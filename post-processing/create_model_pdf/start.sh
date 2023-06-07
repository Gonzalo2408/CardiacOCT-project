~/c-submit --require-cpus=1 --require-mem=30G --priority=low \
 gonzalorodriguez 10160 24 doduo2.umcn.nl/grodriguez/converter:model_pdf \
 --preds /mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/predicted_results_model4_2d_with_maps \
 --pdf_name model4

 #--gpu-count=1 --require-gpu-mem=11g