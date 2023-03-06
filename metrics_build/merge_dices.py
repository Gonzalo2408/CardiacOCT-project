import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import json

def merge_frames_into_pullbacks(path_predicted):

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


path = r'Z:\grodriguez\CardiacOCT\predicted_results_model3_2d'
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')
merged_pullbacks = merge_frames_into_pullbacks(path)
num_classes = 13
final_dict = {}

for pullback in merged_pullbacks.keys():

    # key_patient = pullback.split('_')[0]
    # first_part = key_patient[:3]
    # second_part = key_patient[3:-4]
    # third_part = key_patient[-4:]  
    # patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

    # #Take pullback name
    # n_pullback = pullback.split('_')[1]
    # pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

    print('Pullback ', pullback)

    dices_dict = {}

    for label in range(num_classes):

        print('Checking label ', label)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for frame in merged_pullbacks[pullback]:

            seg_map_data_pred = sitk.GetArrayFromImage(sitk.ReadImage(r'Z:\grodriguez\CardiacOCT\predicted_results_model3_2d\{}'.format(frame)))[0]
            seg_map_data_orig = sitk.GetArrayFromImage(sitk.ReadImage(r'Z:\grodriguez\CardiacOCT\data-2d\nnUNet_raw_data\Task503_CardiacOCT\labelsTs\{}'.format(frame)))[0]

            rows, cols = seg_map_data_orig.shape

            for i in range(rows):
                for j in range(rows):

                    if seg_map_data_pred[i, j] == label and seg_map_data_orig[i, j] == label:
                        tp += 1

                    elif seg_map_data_pred[i, j] == label and seg_map_data_orig[i, j] != label:
                        fp += 1
                        
                    elif seg_map_data_pred[i, j] != label and seg_map_data_orig[i, j] == label:
                        fn += 1

                    elif seg_map_data_pred[i, j] != label and seg_map_data_orig[i, j] != label:
                        tn += 1

        try:
            dice_label = 2*tp / (tp + fp + tp + fn)

        except ZeroDivisionError:
            dice_label = "NaN"

        dices_dict[str(label)] = dice_label 

    final_dict[pullback] = dices_dict

with open('./new_dices_test_silvan.json', 'w') as f:
    json.dump(final_dict, f, indent=4)