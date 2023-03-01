import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

files_path_backup = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNetData_plans_v2.1_stage1'
files_path = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_preprocessed/Task503_CardiacOCT/nnUNetData_plans_v2.1_stage1'

files = os.listdir(files_path)
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

for file in files:

    if file.split('.')[1] == 'pkl':
        continue

    else:
        print('Processing', file)
        n_pullback = file.split('_')[1]
        preprocessed_img = dict(np.load(files_path + '/' + file))
        img_pixel_data = preprocessed_img['data']

        seg_map = img_pixel_data[3,:,:,:]

        frames_with_annot = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == files[0].split('_')[0])]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        print(frames_list)

        for frame in range(len(seg_map)):

            if frame in frames_list:
                continue
        
            else:
                seg_map[frame,:,:] = -1

        img_pixel_data[3,:,:,:] = seg_map

        preprocessed_img['data'] = img_pixel_data

        np.savez_compressed(files_path_backup + '/' + file, **preprocessed_img)
