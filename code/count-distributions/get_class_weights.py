import os
import SimpleITK as sitk
import numpy as np

#Folder with dataset labels
segs_path = r'Z:\grodriguez\CardiacOCT\data-2d\nnUNet_raw_data\Task601_CardiacOCT\labelsTr'
segs_folder = os.listdir(segs_path)

num_classes = 13
label_counts = np.zeros(num_classes)

print('Counting labels...')

class_count = []

for file in segs_folder:

    seg = sitk.ReadImage(os.path.join(segs_path, file))
    seg_data = sitk.GetArrayFromImage(seg)

    #Count the nº of pixels for a label 
    unique, counts = np.unique(seg_data, return_counts=True)
    label_counts[unique] += counts
        
#Get class weight in terms of frequency in the dataset
total_pixels = 704 * 704 * 1810
class_weights = total_pixels / (13 * label_counts)

print("Class weights:")
for label, weight in enumerate(class_weights):
    print(f"Label {label}: {weight}")
