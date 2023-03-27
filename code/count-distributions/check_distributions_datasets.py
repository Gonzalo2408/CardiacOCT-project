import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import os
import sys

sys.path.insert(1, 'Z:/grodriguez/CardiacOCT/post-processing')
from output_handling import create_annotations

def count_pullback_excel(path, segs_folder, excel_name):

    counts_per_pullback = pd.DataFrame(columns = ['pullback', 'dataset', 'set', 'background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                    'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                    'rupture'])

    for file in segs_folder:

        print('Counting in image ', file)

        seg_map = sitk.ReadImage(path + '/' + file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)

        #Some string manipulations to get each variable for a pullback
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        pullback_name = file.split('.')[0]
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]
        dataset = annots.loc[annots['Patient'] == patient_name]['Dataset'].values[0]

        one_hot = np.zeros(num_classes)

        for frame in range(len(seg_map_data)):

            #Add labels that occur in a frame, and get the total count for a pullback
            if frame in frames_list:
                unique, _ = np.unique(seg_map_data[frame,:,:], return_counts=True)  
                unique = unique.astype(int)

                one_hot[[unique[i] for i in range(len(unique))]] += 1

            else:
                continue

        #Add extra variables for each pullback count
        one_hot_list = one_hot.tolist()
        one_hot_list.insert(0, pullback_name)
        one_hot_list.insert(1, dataset)
        one_hot_list.insert(2, belonging_set)

        counts_per_pullback = counts_per_pullback.append(pd.Series(one_hot_list, index=counts_per_pullback.columns[:len(one_hot_list)]), ignore_index=True)

    #Create Excel with results
    counts_per_pullback.to_excel('./{}.xlsx'.format(excel_name))


def count_frames_excel(path, segs_folder, excel_name):

    counts_per_frame = pd.DataFrame(columns = ['pullback', 'dataset', 'set', 'frame',  'background',
                                               'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 'media',
                                               'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                               'rupture', 'cap_thickness', 'lipid_arc'])

    for file in segs_folder:

        print('Counting in image ', file)

        seg_map = sitk.ReadImage(path + '/' + file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)

        #String manipulations as before
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        pullback_name = file.split('.')[0]
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]
        dataset = annots.loc[annots['Patient'] == patient_name]['Dataset'].values[0]

        if belonging_set == 'Training':
            continue
    
        else:

            one_hot = np.zeros(num_classes)

            for frame in range(len(seg_map_data)):

                #Add labels that occur in each case (in this case, it's either 0 or 1, becuase we are doing it per frame)
                if frame in frames_list:
                    unique, _ = np.unique(seg_map_data[frame,:,:], return_counts=True)
                    unique = unique.astype(int)

                    one_hot[[unique[i] for i in range(len(unique))]] = 1

                    #Get post-processing measurements for the specific frame
                    post_image_array , _ , cap_thickness, lipid_arc = create_annotations(seg_map_data[frame,:,:])

                    #Append important variables for each frame
                    one_hot_list = one_hot.tolist()
                    one_hot_list.insert(0, pullback_name)
                    one_hot_list.insert(1, dataset)
                    one_hot_list.insert(2, belonging_set)
                    one_hot_list.insert(3, frame)
                    one_hot_list.append(cap_thickness)
                    one_hot_list.append(lipid_arc)

                    counts_per_frame = counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)

                    post_image_array = np.uint8(post_image_array*255)

                    #Only save images that contain lipid
                    if not np.any(post_image_array) or belonging_set == 'Training':
                        continue

                    else:
                        #Save segmentation with lipid arc and FCT
                        color_map = {
                            0: (0, 0, 0),
                            1: (255, 0, 0),      # red
                            2: (0, 255, 0),      # green
                            3: (0, 0, 255),      # blue
                            4: (255, 255, 0),    # yellow
                            5: (255, 0, 255),    # magenta
                            6: (0, 255, 255),    # cyan
                            7: (128, 0, 0),      # maroon
                            8: (0, 128, 0),      # dark green
                            9: (0, 0, 128),      # navy
                            10: (128, 128, 0),   # olive
                            11: (128, 0, 128),   # purple
                            12: (0, 128, 128),   # teal
                        }

                        #Convert the labels array into a color-coded image
                        h, w = seg_map_data[frame,:,:].shape
                        color_img = np.zeros((h, w, 3), dtype=np.uint8)
                        for label, color in color_map.items():
                            color_img[seg_map_data[frame,:,:] == label] = color
                        seg_image = Image.fromarray(color_img)

                        #Overlay image  
                        post_proc_image = Image.fromarray(post_image_array)
                    
                        seg_image.paste(post_proc_image, (0,0), post_proc_image)
                        seg_image.save('Z:/grodriguez/CardiacOCT/post-processing/post-proc-imgs-orig/{}_frame{}.png'.format(pullback_name, frame))

                else:
                    continue

    #Create Excel file
    #counts_per_frame.to_excel('./{}.xlsx'.format(excel_name))

if __name__ == "__main__":

    num_classes = 13

    path = 'Z:/grodriguez/CardiacOCT/data-original/segmentations-ORIGINALS'
    excel_name = 'new_automatic_measures'

    seg_files = os.listdir(path)
    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

    #count_frames_excel(path_third, seg_files_3, excel_name1)
    count_frames_excel(path, seg_files, excel_name)

