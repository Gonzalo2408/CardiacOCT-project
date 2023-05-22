~/c-submit --gpu-count=1 --require-gpu-mem=10G --require-cpus=2 --require-mem=28g --priority=low \
    gonzalorodriguez 10160 24 doduo1.umcn.nl/grodriguez/converter:imgs_pseudo3d \
    --data /mnt/netcache/diag/grodriguez/CardiacOCT/data-original/extra-scans-DICOM-3    \
    --task Task509_CardiacOCT \
    --k 4