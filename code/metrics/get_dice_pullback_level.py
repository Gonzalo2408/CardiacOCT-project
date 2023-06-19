import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import json
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import sys
sys.path.append("..") 
from utils.counts_utils import merge_frames_into_pullbacks
from utils.metrics_utils import calculate_confusion_matrix, dice_from_cm


path = r'Z:\grodriguez\CardiacOCT\preds-test-set\model8_preds'
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final.xlsx')
json_file_name = 'model8_pullback_level'

merged_pullbacks = merge_frames_into_pullbacks(path)

num_classes = 13
final_dict = {}


for pullback in merged_pullbacks.keys():

    #Get patient name
    key_patient = pullback.split('_')[0]
    first_part = key_patient[:3]
    second_part = key_patient[3:-4]
    third_part = key_patient[-4:]  
    patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

    #Take pullback name
    n_pullback = pullback.split('_')[1]
    pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

    print('Pullback ', pullback)

    dices_dict = {}

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int)

    for frame in merged_pullbacks[pullback]:

        #Load original and pred segmentation
        seg_map_data_pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, frame)))[0]
        seg_map_data_orig = sitk.GetArrayFromImage(sitk.ReadImage('Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task512_CardiacOCT/labelsTs/{}'.format(frame)))[0]

        #Sum cm for each frame so at the end we get the CM for the whole pullback
        cm = calculate_confusion_matrix(seg_map_data_orig, seg_map_data_pred, range(13))       
        cm_total += cm

    dice = dice_from_cm(cm_total)

    #Create dictionary with the labels and DICEs
    for label in range(num_classes):

        #Check for nans and put them in string-like
        if math.isnan(dice[label]):
            dices_dict[str(label)] = "NaN"

        else:

            dices_dict[str(label)] = dice[label] 

    final_dict[pullback_name] = dices_dict


with open("Z:/grodriguez/CardiacOCT/info-files/metrics/{}.json".format(json_file_name), 'w') as f:
    json.dump(final_dict, f, indent=4)