~/c-submit --require-cpus=2 --require-mem=40g --priority=high \
    gonzalorodriguez 10160 24 doduo1.umcn.nl/grodriguez/converter:imgs_pseudo3d \
    --data /mnt/netcache/diag/grodriguez/CardiacOCT/data-original/extra-scans-DICOM-3    \
    --task Task513_CardiacOCT \
    --k 7 \
    --grayscale True

#--gpu-count=1 --require-gpu-mem=11g