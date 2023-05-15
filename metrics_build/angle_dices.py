import SimpleITK as sitk
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

sys.path.insert(1, 'Z:/grodriguez/CardiacOCT/post-processing')
from output_handling import create_annotations_lipid, create_annotations_calcium, compute_new_dices

def merge_frames_into_pullbacks(path_predicted):

    pullbacks_origs = [i for i in os.listdir(path_predicted) if '.nii.gz' in i]
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

def get_dice_frame_level(orig_path, pred_path, excel_name, region):

    orig_segs = os.listdir(orig_path)

    new_excel_data = pd.DataFrame(columns=['pullback', 'frame', 'DICE'])

    for seg in orig_segs:

        list_data = []

        if 'NLDAMPH0028' not in seg:
            continue

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
        orig_img = sitk.ReadImage(os.path.join(orig_path, seg))
        orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

        pred_img = sitk.ReadImage(os.path.join(pred_path, seg))
        pred_img_data = sitk.GetArrayFromImage(pred_img)[0]

        #Get IDs for both cases
        if region == 'lipid':
            _, _, _, _, orig_ids = create_annotations_lipid(orig_img_data)
            _, _, _, _, pred_ids = create_annotations_lipid(pred_img_data)

        elif region == 'calcium':
            _, _, _, _, _, orig_ids = create_annotations_calcium(orig_img_data)
            _, _, _, _, _, pred_ids = create_annotations_calcium(pred_img_data)

        else:
            raise ValueError('Please, select a valid type (lipid or calcium)')

        #Compute new DICE for lipid
        dice_score, _, _, _ = compute_new_dices(orig_ids, pred_ids)

        list_data.append(pullback_name)
        list_data.append(n_frame)
        list_data.append(dice_score)

        new_excel_data = new_excel_data.append(pd.Series(list_data, index=new_excel_data.columns[:len(list_data)]), ignore_index=True)

    new_excel_data.to_excel('./{}.xlsx'.format(excel_name))


def get_dice_pullback_level(orig_path, pred_path, excel_name, region):

    pullback_dict = merge_frames_into_pullbacks(pred_path)
    new_excel_data = pd.DataFrame(columns=['pullback', 'DICE'])

    for pullback in pullback_dict.keys():

        print('Pullback ', pullback)

        #In order to get DICEs pullback-level, we obtain all of the bin IDs for lipid in every frame with annotation in a pullback
        list_data = []

        #Obtain format of pullback name as in the beginning
        filename = pullback.split('_')[0]
        first_part = filename[:3]
        second_part = filename[3:-4]
        third_part = filename[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Obtain pullback name
        n_pullback = pullback.split('_')[1]
        pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

        tp_total = 0
        fp_total = 0
        fn_total = 0

        for frame in pullback_dict[pullback]:

            print('Checking frame ', frame)
            
            orig_img = sitk.ReadImage(os.path.join(orig_path, frame))
            orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

            pred_img = sitk.ReadImage(os.path.join(pred_path, frame))
            pred_img_data = sitk.GetArrayFromImage(pred_img)[0]

            #Get IDs for both cases
            if region == 'lipid':
                _, _, _, _, orig_ids = create_annotations_lipid(orig_img_data)
                _, _, _, _, pred_ids = create_annotations_lipid(pred_img_data)

            elif region == 'calcium':
                _, _, _, _, _, orig_ids = create_annotations_calcium(orig_img_data)
                _, _, _, _, _, pred_ids = create_annotations_calcium(pred_img_data)

            else:
                raise ValueError('Please, select a valid type (lipid or calcium)')

            #Sum all the TP, TN, FP, FN over a full pullback to get the DICE per pullback
            _, tp, fp, fn = compute_new_dices(orig_ids, pred_ids)
            tp_total += tp
            fp_total += fp
            fn_total += fn

        #Compute new DICE for lipids in a pullback
        try:
            dice_score_pullback = 2*tp_total / (tp_total + fp_total + tp_total + fn_total)

        except ZeroDivisionError:
            dice_score_pullback = np.nan 

        print(dice_score_pullback, '\n')

        list_data.append(pullback_name)
        list_data.append(dice_score_pullback)

        new_excel_data = new_excel_data.append(pd.Series(list_data, index=new_excel_data.columns[:len(list_data)]), ignore_index=True)

    new_excel_data.to_excel('./{}.xlsx'.format(excel_name))

if __name__ == "__main__":

    orig_test_segs_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task507_CardiacOCT/labelsTs'
    pred_test_segs_path = 'Z:/grodriguez/CardiacOCT/preds-test-set/predicted_results_model5_pseudo3d_with_maps'

    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')
    excel_file = 'calcium_model5_test_pullback_new_pullback'

    get_dice_pullback_level(orig_test_segs_path, pred_test_segs_path, excel_file, 'calcium')

    #get_dice_frame_level(orig_test_segs_path, pred_test_segs_path, excel_file, 'calcium')
