import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
sys.path.append("..") 
from utils.counts_utils import merge_frames_into_pullbacks
from utils.postprocessing import create_annotations_lipid, create_annotations_calcium, compute_arc_dices


def get_dice_frame_level(orig_path, pred_path, excel_name):
    """Obtain the DICE score for lipid arc and calcium arc on frame level (i.e confusion matrix is computed for every frame)

    Args:
        orig_path (string): path of orginal segmentations (must be the data in labelsTs!!)
        pred_path (string): path of predicted segmentations
        excel_name (string): Excel name to save the results
    """    

    orig_segs = os.listdir(orig_path)

    new_excel_data = pd.DataFrame(columns=['pullback', 'frame', 'DICE lipid', 'DICE calcium'])

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
        orig_img = sitk.ReadImage(os.path.join(orig_path, seg))
        orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

        pred_img = sitk.ReadImage(os.path.join(pred_path, seg))
        pred_img_data = sitk.GetArrayFromImage(pred_img)[0]

        #Get IDs for both cases
        _, _, _, _, orig_ids_lipid = create_annotations_lipid(orig_img_data, font = 'mine')
        _, _, _, _, pred_ids_lipid = create_annotations_lipid(pred_img_data, font = 'mine')

        _, _, _, _, _, orig_ids_calcium = create_annotations_calcium(orig_img_data, font = 'mine')
        _, _, _, _, _, pred_ids_calcium = create_annotations_calcium(pred_img_data, font = 'mine')

        #Compute new DICE for lipid
        dice_score_lipid, _, _, _ = compute_arc_dices(orig_ids_lipid, pred_ids_lipid)
        dice_score_calcium, _, _, _ = compute_arc_dices(orig_ids_calcium, pred_ids_calcium)

        list_data.append(pullback_name)
        list_data.append(n_frame)
        list_data.append(dice_score_lipid)
        list_data.append(dice_score_calcium)

        new_excel_data = new_excel_data.append(pd.Series(list_data, index=new_excel_data.columns[:len(list_data)]), ignore_index=True)

    new_excel_data.to_excel('Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}.xlsx'.format(excel_name))


def get_dice_pullback_level(orig_path, pred_path, excel_name):
    """Obtain the DICE score for lipid arc and calcium arc on pullback level (i.e confusion matrix is computed for the whole pullback)

    Args:
        orig_path (string): path of orginal segmentations (must be the data in labelsTs!!)
        pred_path (string): path of predicted segmentations
        excel_name (string): Excel name to save the results
    """      

    pullback_dict = merge_frames_into_pullbacks(pred_path)
    new_excel_data = pd.DataFrame(columns=['pullback', 'DICE lipid', 'DICE calcium'])

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

        tp_total_lipid = 0
        fp_total_lipid = 0
        fn_total_lipid = 0

        tp_total_calcium = 0
        fp_total_calcium = 0
        fn_total_calcium = 0

        
        for frame in pullback_dict[pullback]:

            print('Checking frame ', frame)
            
            orig_img = sitk.ReadImage(os.path.join(orig_path, frame))
            orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

            pred_img = sitk.ReadImage(os.path.join(pred_path, frame))
            pred_img_data = sitk.GetArrayFromImage(pred_img)[0]

            #Get IDs for both cases
            _, _, _, _, orig_ids_lipid = create_annotations_lipid(orig_img_data, font = 'mine')
            _, _, _, _, pred_ids_lipid = create_annotations_lipid(pred_img_data, font = 'mine')

            _, _, _, _, _, orig_ids_calcium = create_annotations_calcium(orig_img_data, font = 'mine')
            _, _, _, _, _, pred_ids_calcium = create_annotations_calcium(pred_img_data, font = 'mine')


            #Sum all the TP, TN, FP, FN over a full pullback to get the DICE per pullback
            _, tp_lipid, fp_lipid, fn_lipid = compute_arc_dices(orig_ids_lipid, pred_ids_lipid)
            tp_total_lipid += tp_lipid
            fp_total_lipid += fp_lipid
            fn_total_lipid += fn_lipid

            _, tp_calcium, fp_calcium, fn_calcium = compute_arc_dices(orig_ids_calcium, pred_ids_calcium)
            tp_total_calcium += tp_calcium
            fp_total_calcium += fp_calcium
            fn_total_calcium += fn_calcium

        #Compute new DICE for lipids in a pullback
        try:
            dice_score_pullback_lipid = 2*tp_total_lipid / (tp_total_lipid + fp_total_lipid + tp_total_lipid + fn_total_lipid)

        except ZeroDivisionError:
            dice_score_pullback_lipid = np.nan 

        try:
            dice_score_pullback_calcium = 2*tp_total_calcium / (tp_total_calcium + fp_total_calcium + tp_total_calcium + fn_total_calcium)

        except ZeroDivisionError:
            dice_score_pullback_calcium = np.nan 

        print('Lipid', dice_score_pullback_lipid)
        print('Calcium', dice_score_pullback_calcium, '\n')

        list_data.append(pullback_name)
        list_data.append(dice_score_pullback_lipid)
        list_data.append(dice_score_pullback_calcium)

        new_excel_data = new_excel_data.append(pd.Series(list_data, index=new_excel_data.columns[:len(list_data)]), ignore_index=True)

    new_excel_data.to_excel('Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}.xlsx'.format(excel_name))

if __name__ == "__main__":

    orig_test_segs_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task601_CardiacOCT/labelsTs'
    pred_test_segs_path = 'Z:/grodriguez/CardiacOCT/preds_second_split/model_rgb_2d_preds'

    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')
    excel_file = 'rgb_2d_frame'

    #get_dice_pullback_level(orig_test_segs_path, pred_test_segs_path, excel_file)

    get_dice_frame_level(orig_test_segs_path, pred_test_segs_path, excel_file)
