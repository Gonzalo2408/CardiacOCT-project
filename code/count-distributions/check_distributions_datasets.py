import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def count_pullback_excel(path, segs_folder, excel_name):

    counts_per_pullback = pd.DataFrame(columns = ['pullback', 'set', 'background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                    'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                    'rupture'])

    for file in segs_folder:

        print('Counting in image ', file)

        seg_map = sitk.ReadImage(path + '/' + file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)

        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        pullback_name = file.split('.')[0]
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        one_hot = np.zeros(num_classes)

        for frame in range(len(seg_map_data)):

            if frame in frames_list:
                unique, _ = np.unique(seg_map_data[frame,:,:], return_counts=True)  
                unique = unique.astype(int)

                one_hot[[unique[i] for i in range(len(unique))]] += 1

            else:
                continue

        one_hot_list = one_hot.tolist()
        one_hot_list.insert(0, pullback_name)
        one_hot_list.insert(1, belonging_set)

        counts_per_pullback = counts_per_pullback.append(pd.Series(one_hot_list, index=counts_per_pullback.columns[:len(one_hot_list)]), ignore_index=True)
    counts_per_pullback.to_excel('./{}.xlsx'.format(excel_name))


def count_frames_excel(path, segs_folder, excel_name):

    counts_per_frame = pd.DataFrame(columns = ['pullback', 'dataset', 'set', 'frame',  'background', 
                                               'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 'media', 
                                               'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                               'rupture'])

    for file in segs_folder:

        print('Counting in image ', file)

        seg_map = sitk.ReadImage(path + '/' + file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)

        #Get frame values (patient, pullback, set and first or second dataset)
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        pullback_name = file.split('.')[0]
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]
        dataset = annots.loc[annots['Patient'] == patient_name]['Dataset'].values[0]

        one_hot = np.zeros(num_classes)

        for frame in range(len(seg_map_data)):

            if frame in frames_list:
                unique, _ = np.unique(seg_map_data[frame,:,:], return_counts=True)  
                unique = unique.astype(int)

                one_hot[[unique[i] for i in range(len(unique))]] = 1

            else:
                continue

            one_hot_list = one_hot.tolist()
            one_hot_list.insert(0, pullback_name)
            one_hot_list.insert(1, dataset)
            one_hot_list.insert(2, belonging_set)
            one_hot_list.insert(3, frame)
            

            counts_per_frame = counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)
    counts_per_frame.to_excel('./{}.xlsx'.format(excel_name))

if __name__ == "__main__":

    num_classes = 13

    path_first = 'Z:/grodriguez/CardiacOCT/data-original/segmentations ORIGINALS'
    path_sec = 'Z:/grodriguez/CardiacOCT/data-original/extra segmentations ORIGINALS'
    excel_name = ''

    seg_files_1 = os.listdir(path_first)
    seg_files_2 = os.listdir(path_sec)
    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

    #count_frames_excel(path_sec, seg_files_2, excel_name)
    #count_pullback_excel(path_sec, seg_files_2, excel_name)

    