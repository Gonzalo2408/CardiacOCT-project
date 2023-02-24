import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import json

num_classes = 13

#Count among frames
def build_excel_frames(path_dir, segs_dir, excel_name):

    counts_per_frame = pd.DataFrame(columns = ['pullback', 'frame', 'background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                'rupture', 'set'])


    for file in segs_dir:

        seg_map = sitk.ReadImage(path_dir + '/' + file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)

        #Undo filename (add hyphens)
        filename = file.split('_')[0]
        first_part = filename[:3]
        second_part = filename[3:-4]
        third_part = filename[-4:]  
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Obtain pullback name
        n_pullback = file.split('_')[1]
        pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

        #Obtain nº frame
        n_frame = file.split('_')[2][5:]

        #frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        #frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        one_hot = np.zeros(num_classes)

        unique, _ = np.unique(seg_map_data, return_counts=True)  
        unique = unique.astype(int)

        one_hot[[unique[i] for i in range(len(unique))]] = 1

        one_hot_list = one_hot.tolist()
        one_hot_list.insert(0, pullback_name)
        one_hot_list.insert(1, n_frame)
        one_hot_list.append(belonging_set)

        counts_per_frame = counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)

    counts_per_frame.to_excel('./{}.xlsx'.format(excel_name))

#Count among pullbacks
def build_excel_pullacks(path_dir, segs_dir, excel_name):

    counts_per_pullback = pd.DataFrame(columns = ['pullback','background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                'rupture', 'set'])

    


if __name__ == "__main__":
    #For train cases
    # path_train_res_first = 'Z:/grodriguez/CardiacOCT/data-2d/results/nnUNet/2d/Task501_CardiacOCT/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_postprocessed'
    # seg_files_res_train_1 = sorted(os.listdir(path_train_res_first))
    # seg_files_res_train_1.pop()

    #For test cases
    path_test_res_first = 'Z:/grodriguez/CardiacOCT/predicted_results_model1_2d'
    seg_files_res_test_1 = sorted(os.listdir(path_test_res_first))[:-3]

    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

    build_excel_frames(path_test_res_first, seg_files_res_test_1, 'counts_test_first')