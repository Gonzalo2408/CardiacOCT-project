import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import pandas as pd
import warnings
import math
import argparse
import sys
import pandas as pd
sys.path.append("..") 
from utils.metrics_utils import calculate_confusion_matrix, metrics_from_cm
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_prob_maps_list(prob_map):

    #Takes the npz file with the prob maps and creates an array of shape (num_classes, img_shape)
    num_classes = 13
    probs_list = []

    for i in prob_map.items():

        for label in range(num_classes):
            probs_list.append(i[1][label][0])

    #WTF prob map
    if probs_list[0].shape[0] == 690:
        prob_img = np.zeros((num_classes, 690, 691))

    else:   
        prob_img = np.zeros((num_classes, 691, 691))

    _, rows, cols = prob_img.shape

    for i in range(rows):
        for j in range(cols):

            prob_img[:, i, j] = np.array([probs_list[label][i, j] for label in range(num_classes)])

    return prob_img

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=str, default='Z:/grodriguez/CardiacOCT/preds_second_split/model_rgb_2d_preds')
    parser.add_argument('--excel_name', type=str, default='model_rgb_2d')
    args, _ = parser.parse_known_args(argv)

    orig_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task602_CardiacOCT/labelsTs'

    orig_list = os.listdir(orig_path)

    df = pd.DataFrame(columns = ['Pullback', 'Frame', 'CM', 'DICE', 'Lipid prob'])

    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')

    for orig in orig_list:

        print('Checking ', orig)

        #Obtain format of pullback name (it's different than in the dataset counting)
        filename = orig.split('_')[0]
        first_part = filename[:3]
        second_part = filename[3:-4]
        third_part = filename[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Obtain pullback name
        n_pullback = orig.split('_')[1]
        pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

        #Obtain nº frame
        n_frame = orig.split('_')[2][5:]

        #Get orig array
        orig_seg = sitk.ReadImage(os.path.join(orig_path, orig))
        orig_seg_data = sitk.GetArrayFromImage(orig_seg)[0]

        #Get pred array
        pred_seg = sitk.ReadImage(os.path.join(args.preds_path, orig))
        pred_seg_data = sitk.GetArrayFromImage(pred_seg)[0]

        #Get prob map from npz files
        prob_map = np.load(os.path.join(args.preds_path, '{}.npz'.format(orig.split('.')[0])))

        prob_img = get_prob_maps_list(prob_map)
        prob_img_max = np.max(prob_img, axis=0)

        #For weird case of missing pixel
        if prob_img_max.shape == (691,691):

            true_seg_crop = orig_seg_data[6:697, 6:697]
            pred_seg_crop = pred_seg_data[6:697, 6:697]

        else:

            true_seg_crop = orig_seg_data[6:696, 6:697]
            pred_seg_crop = pred_seg_data[6:696, 6:697]


        #Get for all lipid pixels the probability of lipid
        orig_lipid = np.mean(prob_img_max[true_seg_crop == 4])
        pred_lipid = np.mean(prob_img_max[pred_seg_crop == 4])

        cm = calculate_confusion_matrix(true_seg_crop, pred_seg_crop, range(13))
        dice_lipid = metrics_from_cm(cm)[0][4]

        if math.isnan(orig_lipid) and math.isnan(pred_lipid):

            prob = np.nan 
            cm_type = 'TN'

        elif math.isnan(orig_lipid) == True and math.isnan(pred_lipid) == False:

            prob = pred_lipid
            cm_type = 'FP'

        elif math.isnan(orig_lipid) == False and math.isnan(pred_lipid) == True:

            prob = orig_lipid
            cm_type = 'FN'

        else:

            prob = pred_lipid
            cm_type = 'TP'


        list_to_df = []
        list_to_df.append(pullback_name)
        list_to_df.append(n_frame)
        list_to_df.append(cm_type)
        list_to_df.append(dice_lipid)
        list_to_df.append(prob)

        df = df.append(pd.Series(list_to_df, index=df.columns[:len(list_to_df)]), ignore_index=True)

    df.to_excel('Z:/grodriguez/CardiacOCT/code/uncertainty-estimation/{}.xlsx'.format(args.excel_name))

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)




    

    

