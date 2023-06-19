import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

new_npz_path = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNetData_plans_v2.1_stage1'
npz_path = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_preprocessed/Task504_CardiacOCT/nnUNetData_plans_v2.1_stage1'

new_cropped_npz_path = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNetData_plans_v2.1_stage1_cropped'
cropped_npz_path = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_cropped_data/Task504_CardiacOCT'

new_gt_segs_path = 'Z:/grodriguez/CardiacOCT/data-3d/gt_segmentations'
gt_segs_path = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_preprocessed/Task504_CardiacOCT/gt_segmentations'


files_npz = os.listdir(npz_path)
files_npz_cropped = os.listdir(cropped_npz_path)
files_gt = os.listdir(gt_segs_path)
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final.xlsx')

print('Changing npz files')
for file in files_npz:

    filename = file.split('.')

    if len(filename) == 1:
        continue

    else:
        if file.split('.')[1] == 'npz':

            print('Processing', file)
            n_pullback = file.split('_')[1]
            patient_name = file.split('_')[0]
            preprocessed_img = dict(np.load(npz_path + '/' + file))
            img_pixel_data = preprocessed_img['data']

            seg_map = np.zeros((img_pixel_data.shape[1], img_pixel_data.shape[2], img_pixel_data.shape[3]))

            frames_with_annot = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Frames']
            frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
            print(frames_list)

            for frame in range(img_pixel_data.shape[3]):

                if frame in frames_list:
                    seg_map[:,:,frame] = img_pixel_data[3,:,:,frame]
            
                else:
                    seg_map[:,:,frame] = -1

            img_pixel_data[3,:,:,:] = seg_map

            preprocessed_img['data'] = img_pixel_data

            np.savez_compressed(new_npz_path + '/' + file, **preprocessed_img)

        else:
            continue