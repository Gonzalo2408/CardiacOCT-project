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


def count_frames_excel_dataset(path, excel_name, annots):

    """Creates Excel file with the label count and lipid and calcium measurements 
       for every frame that contains annotations in the dataset (these are then copied into the counts Excel)

    Args:
        path (string): segmentations path 
        excel_name (string): name of the Excel file to be created
        annots (dataframe): dataframe containing all the "metadata" for every pullback in the dataset
    """    

    counts_per_frame = pd.DataFrame(columns = ['pullback', 'dataset', 'set', 'frame',  'background',
                                               'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 'media',
                                               'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                               'rupture', 'cap_thickness', 'lipid_arc', 'calcium_depth', 'calcium_arc', 'calcium_thickness'])
    
    for file in os.listdir(path):

        #See only nifti files
        if file.endswith('nii.gz') == False:
            continue

        print('Counting in image ', file)

        seg_map = sitk.ReadImage(path + '/' + file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)

        #Get data from the file to be processed
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        pullback_name = file.split('.')[0]
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]
        dataset = annots.loc[annots['Patient'] == patient_name]['Dataset'].values[0]

        num_classes = 13

        for frame in range(len(seg_map_data)):

            #Add labels that occur in each case (in this case, it's either 0 or 1)
            if frame in frames_list:

                one_hot = np.zeros(num_classes)

                unique, _ = np.unique(seg_map_data[frame,:,:], return_counts=True)
                unique = unique.astype(int)

                #Get final array with 0 and 1 for a frame
                one_hot[[unique[i] for i in range(len(unique))]] = 1

                #Get post-processing measurements for the specific frame
                _, _ , cap_thickness, lipid_arc, _ = create_annotations_lipid(seg_map_data[frame,:,:], font='mine')
                _, _, calcium_depth, calcium_arc, calcium_thickness, _ = create_annotations_calcium(seg_map_data[frame,:,:], font='mine')

                #Append important variables that we want to display in the Excel file
                one_hot_list = one_hot.tolist()
                one_hot_list.insert(0, pullback_name)
                one_hot_list.insert(1, dataset)
                one_hot_list.insert(2, belonging_set)
                one_hot_list.insert(3, frame)
                one_hot_list.append(cap_thickness)
                one_hot_list.append(lipid_arc)
                one_hot_list.append(calcium_depth)
                one_hot_list.append(calcium_arc)
                one_hot_list.append(calcium_thickness)

                counts_per_frame = counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)

            else:
                continue

    #Create Excel file
    counts_per_frame.to_excel('Z:/grodriguez/CardiacOCT/info-files/counts/{}.xlsx'.format(excel_name))

if __name__ == "__main__":

    path = 'Z:/grodriguez/CardiacOCT/data-original/ISALAS'
    excel_name = 'isalas'

    seg_files = os.listdir(path)
    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')

    count_frames_excel_dataset(path, excel_name, annots)

