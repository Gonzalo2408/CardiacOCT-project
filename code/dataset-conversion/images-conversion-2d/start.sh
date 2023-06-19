~/c-submit --require-cpus=1 --require-mem=32G --priority=low \
 gonzalorodriguez 10160 24 doduo2.umcn.nl/grodriguez/converter:imgs2_2d \
 --data /mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans-DICOM \
 --task Task999_CardiacOCT \
 --grayscale True

 #--gpu-count=1 --require-gpu-mem=10G