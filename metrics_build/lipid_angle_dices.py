import SimpleITK as sitk
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

sys.path.insert(1, 'Z:/grodriguez/CardiacOCT/post-processing')
from output_handling import create_annotations, compute_new_dices

orig_test_segs_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task503_CardiacOCT/labelsTs'
pred_test_segs_path = 'Z:/grodriguez/CardiacOCT/predicted_results_model1_2d'

orig_segs = os.listdir(orig_test_segs_path)
pred_segs = os.listdir(pred_test_segs_path)[:-3]

annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')
excel_file = 'lipid_angle_dices_model1'

new_excel_data = pd.DataFrame(columns=['pullback', 'frame', 'DICE'])

for seg in orig_segs:

    list_data = []

    print('Case ', seg)

    #Obtain format of pullback name as in the beginning
    filename = seg.split('_')[0]
    first_part = filename[:3]
    second_part = filename[3:-4]
    third_part = filename[-4:]
    patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

    #Obtain pullback name
    n_pullback = seg.split('_')[1]
    pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

    #Obtain nº frame and set (train/test)
    n_frame = seg.split('_')[2][5:]

    #Read original and pred segmentation
    orig_img = sitk.ReadImage(os.path.join(orig_test_segs_path, seg))
    orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

    pred_img = sitk.ReadImage(os.path.join(pred_test_segs_path, seg))
    pred_img_data = sitk.GetArrayFromImage(pred_img)[0]

    #Get lipid IDs for both cases
    _, _, _, _, orig_lipid_ids = create_annotations(orig_img_data)
    _, _, _, _, pred_lipid_ids = create_annotations(pred_img_data)

    #Compute new DICE for lipid
    dice_score = compute_new_dices(orig_lipid_ids, pred_lipid_ids)

    list_data.append(pullback_name)
    list_data.append(n_frame)
    list_data.append(dice_score)

    new_excel_data = new_excel_data.append(pd.Series(list_data, index=new_excel_data.columns[:len(list_data)]), ignore_index=True)

new_excel_data.to_excel('./{}.xlsx'.format(excel_file))

