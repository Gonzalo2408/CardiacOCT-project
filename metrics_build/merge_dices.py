import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def merge_frames_into_pullbacks(path_predicted):

    pullbacks_origs = [file for file in os.listdir(path_predicted) if file.endswith('nii.gz')]
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
    # keys = list(pullbacks_dict.keys())[-3:]
    # for key in keys:
    #     pullbacks_dict[key].pop()
    #     if not pullbacks_dict[key]:
    #         pullbacks_dict.pop(key)

    return pullbacks_dict

def calculate_confusion_matrix(Y_true, Y_pred, labels):

    cm = np.zeros((len(labels), len(labels)), dtype=np.int)

    for i, x in enumerate(labels):
        for j, y in enumerate(labels):

            cm[i, j] = np.sum((Y_true == x) & (Y_pred == y))

    return cm

def dice_from_cm(cm):

    assert (cm.ndim == 2)
    assert (cm.shape[0] == cm.shape[1])

    dices = np.zeros((cm.shape[0]))

    for i in range(cm.shape[0]):
        dices[i] = 2 * cm[i, i] / float(np.sum(cm[i, :]) + np.sum(cm[:, i]))

    return dices


path = 'Z:/grodriguez/CardiacOCT/preds-test-set/predicted_results_model7_pseudo3d_with_maps'
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')
merged_pullbacks = merge_frames_into_pullbacks(path)
num_classes = 13
final_dict = {}



for pullback in merged_pullbacks.keys():

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

        seg_map_data_pred = sitk.GetArrayFromImage(sitk.ReadImage('Z:/grodriguez/CardiacOCT/preds-test-set/predicted_results_model7_pseudo3d_with_maps/{}'.format(frame)))[0]
        seg_map_data_orig = sitk.GetArrayFromImage(sitk.ReadImage('Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task508_CardiacOCT/labelsTs/{}'.format(frame)))[0]


        labels_present = np.unique(seg_map_data_orig)
        cm = calculate_confusion_matrix(seg_map_data_orig, seg_map_data_pred, range(13))
        cm_total += cm

    dice = dice_from_cm(cm_total)

    for label in range(num_classes):
        dices_dict[str(label)] = dice[label] 

    final_dict[pullback_name] = dices_dict


with open('./pullback_model7_test_dice.json', 'w') as f:
    json.dump(final_dict, f, indent=4)