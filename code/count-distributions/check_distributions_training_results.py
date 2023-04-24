import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image
sys.path.insert(1, 'Z:/grodriguez/CardiacOCT/post-processing')
from output_handling import create_annotations_lipid, create_annotations_calcium


def merge_frames_into_pullbacks(path_predicted):

    #This function creates a dictionary in which the key is the pullback ID and the keys are each file that belongs to a
    #frame in that pullback

    pullbacks_origs = os.listdir(path_predicted)
    pullbacks_origs_set = []
    pullbacks_dict = {}

    #Save into a list the patiend id + n_pullback substring
    for i in range(len(pullbacks_origs)):
        if pullbacks_origs[i].split('_frame')[0] not in pullbacks_origs_set:
            pullbacks_origs_set.append(pullbacks_origs[i].split('_frame')[0])

        else:
            continue

    #Create dict with patient_id as key and list of belonging frames as values
    for i in range(len(pullbacks_origs_set)):
        frames_from_pullback = [frame for frame in pullbacks_origs if pullbacks_origs_set[i] in frame]
        pullbacks_dict[pullbacks_origs_set[i]] = frames_from_pullback

    #Remove last 3 key-value pairs (they are not frames)
    keys = list(pullbacks_dict.keys())[-3:]
    for key in keys:
        pullbacks_dict[key].pop()
        if not pullbacks_dict[key]:
            pullbacks_dict.pop(key)

    return pullbacks_dict


#Count among frames
def build_excel_frames(path_dir, segs_dir, excel_name, save_image=False):

    counts_per_frame = pd.DataFrame(columns = ['pullback', 'frame', 'set', 'background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                'rupture', 'lipid arc', 'cap_thickness', 'calcium_depth', 'calcium_arc', 'calcium_thickness'])


    for file in segs_dir:

        print('Copying {} ...'.format(file))

        seg_map = sitk.ReadImage(path_dir + '/' + file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)

        #Obtain format of pullback name as in the beginning
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

        #Get count of labels in each frame (note that the segs_dir contains in this case a a file for each frame)
        one_hot = np.zeros(num_classes)

        unique, _ = np.unique(seg_map_data, return_counts=True)
        unique = unique.astype(int)

        one_hot[[unique[i] for i in range(len(unique))]] = 1

        one_hot_list = one_hot.tolist()
        one_hot_list.insert(0, pullback_name)
        one_hot_list.insert(1, n_frame)
        one_hot_list.insert(2, belonging_set)

        #Post-processing results
        post_image_array, _ , cap_thickness, lipid_arc, _ = create_annotations_lipid(seg_map_data[0])
        post_image_array, _ , calcium_depth, calcium_arc, calcium_thickness, _ = create_annotations_calcium(seg_map_data[0])


        one_hot_list.append(lipid_arc)
        one_hot_list.append(cap_thickness)
        one_hot_list.append(calcium_depth)
        one_hot_list.append(calcium_arc)
        one_hot_list.append(calcium_thickness)
        counts_per_frame = counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)

        if save_image == True:

            post_image_array = np.uint8(post_image_array*255)

            #Only save images that contain lipid
            if not np.any(post_image_array):
                continue

            else:
                #Save segmentation with lipid arc and FCT
                color_map = {
                    0: (0, 0, 0),
                    1: (255, 0, 0),      #red
                    2: (0, 255, 0),      #green
                    3: (0, 0, 255),      #blue
                    4: (255, 255, 0),    #yellow
                    5: (255, 0, 255),    #magenta
                    6: (0, 255, 255),    #cyan
                    7: (128, 0, 0),      #maroon
                    8: (0, 128, 0),      #dark green
                    9: (0, 0, 128),      #navy
                    10: (128, 128, 0),   #olive
                    11: (128, 0, 128),   #purple
                    12: (0, 128, 128),   #teal
                }

                #Convert the labels array into a color-coded image
                h, w = seg_map_data[0].shape
                color_img = np.zeros((h, w, 3), dtype=np.uint8)
                for label, color in color_map.items():
                    color_img[seg_map_data[0] == label] = color
                seg_image = Image.fromarray(color_img)

                post_proc_image = Image.fromarray(post_image_array)
            
                #Overlay image
                seg_image.paste(post_proc_image, (0,0), post_proc_image)
                seg_image.save('Z:/grodriguez/CardiacOCT/post-processing/post-proc-imgs-model3-2d/{}_frame{}.png'.format(pullback_name, n_frame))

        else:
            continue

    counts_per_frame.to_excel('./{}.xlsx'.format(excel_name))

#Count among pullbacks
def build_excel_pullbacks(path_dir, excel_name):

    counts_per_pullback = pd.DataFrame(columns = ['pullback', 'set', 'background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                'rupture'])

    pullbacks_dict = merge_frames_into_pullbacks(path_dir)

    for pullback in pullbacks_dict.keys():

        print('Checking pullback ', pullback)

        #Obtain format of pullback name as in the beginning
        filename = pullback.split('_')[0]
        first_part = filename[:3]
        second_part = filename[3:-4]
        third_part = filename[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Get specific pullback we are viewing
        n_pullback = pullback.split('_')[1]
        pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        one_hot = np.zeros(num_classes)

        for file in pullbacks_dict[pullback]:

            seg_map = sitk.ReadImage(path_dir + '/' + file)
            seg_map_data = sitk.GetArrayFromImage(seg_map)

            unique, _ = np.unique(seg_map_data, return_counts=True)
            unique = unique.astype(int)

            one_hot[[unique[i] for i in range(len(unique))]] += 1

        one_hot_list = one_hot.tolist()
        one_hot_list.insert(0, pullback_name)
        one_hot_list.insert(1, belonging_set)

        counts_per_pullback = counts_per_pullback.append(pd.Series(one_hot_list, index=counts_per_pullback.columns[:len(one_hot_list)]), ignore_index=True)

    counts_per_pullback.to_excel('./{}.xlsx'.format(excel_name))

if __name__ == "__main__":

    num_classes = 13
    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')
    #path_preds = 'Z:/grodriguez/CardiacOCT/predicted_results_model4_2d'
    path_preds = r'Z:\grodriguez\CardiacOCT\data-2d\results\nnUNet\2d\Task504_CardiacOCT\nnUNetTrainerV2__nnUNetPlansv2.1\cv_niftis_postprocessed'
    preds_list = sorted(os.listdir(path_preds))[:-1]
    name_excel = 'new_val_pred_measurements_with_cal_model4'

    build_excel_frames(path_preds, preds_list, name_excel)
    #build_excel_pullbacks(path_preds, name_excel)