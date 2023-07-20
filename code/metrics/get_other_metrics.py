import SimpleITK as sitk
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import sys
import math
sys.path.append("..") 
from utils.metrics_utils import calculate_confusion_matrix, metrics_from_cm

path_preds = r'Z:\grodriguez\CardiacOCT\preds_second_split\model_rgb_2d_last'
path_origs = r'Z:\grodriguez\CardiacOCT\data-2d\nnUNet_raw_data\Task601_CardiacOCT\labelsTs'

num_classes = 13

def get_metrics_detection():
    """Obtain precision, recall and specificity based on labels that appear in each frame

    Returns:
        dict: dictionary with this fashion: {label1: {prec: x, recall: x, spec: x}, label2: {prec: x, recall: x, spec: x}, ...}
    """    

    cm_dict = {}

    for file in os.listdir(path_preds):

        if file.endswith('nii.gz') == False:
            continue
        
        else:

            print('Checking case', file)
            orig_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_origs, file)))
            pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_preds, file)))

            #Get labels that apppear in frame
            unique_orig = np.unique(orig_seg)
            unique_pred = np.unique(pred_seg)

            #First, we want to check which labels occur and do not occur in both prediction and original
            for label in range(num_classes):

                tp = 0
                tn = 0
                fp = 0
                fn = 0
                
                if label in unique_orig and label in unique_pred:
                    tp += 1

                elif label not in unique_orig and label not in unique_pred:
                    tn += 1

                elif label not in unique_orig and label in unique_pred:
                    fp += 1

                elif label in unique_orig and label not in unique_pred:
                    fn += 1

                #create dictionary with the CM values for every label
                if label not in cm_dict:
                    cm_dict[label] = [tp, tn, fp, fn]

                else:
                    cm_dict[label][0] += tp
                    cm_dict[label][1] += tn
                    cm_dict[label][2] += fp
                    cm_dict[label][3] += fn

    #Create new dict with metrics using the CM dict
    final_dict = {}
    for label in cm_dict.keys():

        tp_total = cm_dict[label][0]
        tn_total = cm_dict[label][1]
        fp_total = cm_dict[label][2]
        fn_total = cm_dict[label][3]

        metrics_dict = {}
        
        #Check when there are no TP, FP, FN or TN
        try:
            prec = tp_total/(tp_total+fp_total)
        except ZeroDivisionError:
            prec = 'NaN'

        try:
            recall = tp_total/(tp_total+fn_total)
        except ZeroDivisionError:
            recall = 'NaN'

        try:
            spec = tn_total/(tn_total+fp_total)
        except ZeroDivisionError:
            spec = 'NaN'


        metrics_dict['Precision'] = prec
        metrics_dict['Recall'] = recall
        metrics_dict['Specificity'] = spec

        final_dict[str(label)] = metrics_dict

    return final_dict

    
def get_metrics_pixel():
    """Obtain precision, recall and specificity based on pixel level

    Returns:
        dict: dictionary with this fashion: {label1: {prec: x, recall: x, spec: x}, label2: {prec: x, recall: x, spec: x}, ...}
    """    

    cm_total = np.zeros((num_classes, num_classes))

    for file in os.listdir(path_preds):

        if file.endswith('nii.gz') == False:
            continue
        
        else:

            print('Checking case', file)
            orig_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_origs, file)))
            pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_preds, file)))

            #In this case, we look at every pixel in the image (we can do same as in the get_dice_pullback_level)
            cm = calculate_confusion_matrix(orig_seg, pred_seg, range(13))       
            cm_total += cm

    _, prec, recall, spec = metrics_from_cm(cm_total)
    final_dict = {}

    #Create dictionary with the labels and metrics (except DICE)
    for label in range(num_classes):

        metrics_dict = {}

        #Check for NaNs
        if math.isnan(prec[label]):
            metrics_dict['Precision'] = "NaN"

        if math.isnan(recall[label]):
            metrics_dict['Recall'] = "NaN"

        if math.isnan(spec[label]):
            metrics_dict['Specificity'] = "NaN"

        else:

            metrics_dict['Precision'] = prec[label] 
            metrics_dict['Recall'] = recall[label] 
            metrics_dict['Specificity'] = spec[label] 

        final_dict[str(label)] = metrics_dict

    return final_dict

if __name__ == "__main__":

    metrics_pixel = get_metrics_pixel()
    metrics_detection = get_metrics_detection()



