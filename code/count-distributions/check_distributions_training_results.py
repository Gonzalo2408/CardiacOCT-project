import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import SimpleITK as sitk
import os
import sys
sys.path.append("..") 
from utils.postprocessing import create_annotations_lipid, create_annotations_calcium


def count_frames_excel_preds(path_dir, excel_name, annots):

    """Creates Excel file with the label count and lipid and calcium measurements 
       for every frame that contains annotations in the specified predicted folder (these are then copied into the counts Excel)

    Args:
        path (string): segmentations path (predicted data)
        excel_name (string): name of the Excel file to be created
        annots (dataframe): dataframe containing all the "metadata" for every pullback in the dataset
    """    

    counts_per_frame = pd.DataFrame(columns = ['pullback', 'frame', 'set', 'background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                'rupture', 'lipid arc', 'cap_thickness', 'calcium_depth', 'calcium_arc', 'calcium_thickness'])


    for file in os.listdir(path_dir):

        #Check only nifti files
        if file.endswith('nii.gz') == False:
            continue

        else:

            #Obtain format of pullback name (it's different than in the dataset counting)
            filename = file.split('_')[0]
            first_part = filename[:3]
            second_part = filename[3:-4]
            third_part = filename[-4:]
            patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

            #Obtain pullback name
            n_pullback = file.split('_')[1]
            pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

            #Obtain nº frame and set (train/test)
            n_frame = file.split('_')[2][5:]
            belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

            print('Copying {} ...'.format(file))

            seg_map = sitk.ReadImage(path_dir + '/' + file)
            seg_map_data = sitk.GetArrayFromImage(seg_map)

            #Get count of labels in each frame
            one_hot = np.zeros(num_classes)

            unique, _ = np.unique(seg_map_data, return_counts=True)
            unique = unique.astype(int)

            one_hot[[unique[i] for i in range(len(unique))]] = 1

            #Post-processing results
            _, _ , cap_thickness, lipid_arc, _ = create_annotations_lipid(seg_map_data[0], font = 'mine')
            _, _ , calcium_depth, calcium_arc, calcium_thickness, _ = create_annotations_calcium(seg_map_data[0], font = 'mine')

            #Create one hot list with all data
            one_hot_list = one_hot.tolist()
            one_hot_list.insert(0, pullback_name)
            one_hot_list.insert(1, n_frame)
            one_hot_list.insert(2, belonging_set)
            one_hot_list.append(lipid_arc)
            one_hot_list.append(cap_thickness)
            one_hot_list.append(calcium_depth)
            one_hot_list.append(calcium_arc)
            one_hot_list.append(calcium_thickness)
            counts_per_frame = counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)

    counts_per_frame.to_excel('Z:/grodriguez/CardiacOCT/info-files/counts/{}.xlsx'.format(excel_name))

if __name__ == "__main__":

    num_classes = 13
    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final.xlsx')
    #path_preds = r'Z:\grodriguez\CardiacOCT\preds-test-set\model9_preds'
    path_preds = 'Z:/grodriguez/CardiacOCT/data-2d/results/nnUNet/2d/Task513_CardiacOCT/nnUNetTrainer_V2_Loss_CEandDice_Weighted__nnUNetPlansv2.1/cv_niftis_postprocessed'
    excel_name = 'new_frames9_train'

    count_frames_excel_preds(path_preds, excel_name, annots)