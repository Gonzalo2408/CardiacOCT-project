import SimpleITK as sitk
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import sys
import json
import math
sys.path.append("..") 
from utils.metrics_utils import calculate_confusion_matrix, metrics_from_cm

def get_metrics_detection():
    """Obtain PPV, NPV, sensitivity, specificity and cohen's kappa based on labels that appear in each frame

    Returns:
        dict: dictionary with this fashion: {label1: {metric1: x, metric2: x, ...}, label2: {metric1: x, metric2: x, ...}, ...}
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
            ppv = tp_total / (tp_total + fp_total)  #precision
        except ZeroDivisionError:
            ppv = 'NaN'

        try:
            npv = tn_total / (tn_total + fn_total)  
        except ZeroDivisionError:
            npv = 'NaN'

        try:
            sens = tp_total / (tp_total + fn_total) #recall
        except ZeroDivisionError:
            sens = 'NaN'

        try:
            spec = tn_total / (tn_total + fp_total)
        except ZeroDivisionError:
            spec = 'NaN'

        try:
            kappa =  2 * (tp_total*tn_total - fn_total*fp_total) / float((tp_total+fp_total)*(fp_total+tn_total) + (tp_total+fn_total)*(fn_total+tn_total))
        except ZeroDivisionError:
            kappa = 'NaN'


        metrics_dict['PPV'] = ppv
        metrics_dict['NPV'] = npv
        metrics_dict['Sensibility'] = sens
        metrics_dict['Specificity'] = spec
        metrics_dict['Kappa'] = kappa

        final_dict[str(label)] = metrics_dict

    return final_dict

    
def get_metrics_pixel():
    """Obtain PPV, NPV, sensitivity, specificity and cohen's kappa based on pixel level

    Returns:
        dict: dictionary with this fashion: {label1: {metric1: x, metric2: x, ...}, label2: {metric1: x, metric2: x, ...}, ...}
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

    _, ppv, npv, sens, spec, kappa = metrics_from_cm(cm_total)
    final_dict = {}

    #Create dictionary with the labels and metrics (except DICE)
    for label in range(num_classes):

        metrics_dict = {}

        #Check for NaNs
        if math.isnan(ppv[label]):
            metrics_dict['PPV'] = "NaN"

        if math.isnan(npv[label]):
            metrics_dict['NPV'] = "NaN"

        if math.isnan(sens[label]):
            metrics_dict['Sensibility'] = "NaN"

        if math.isnan(spec[label]):
            metrics_dict['Specificity'] = "NaN"

        if math.isnan(kappa[label]):
            metrics_dict['Kappa'] = "NaN"

        else:

            metrics_dict['PPV'] = ppv[label] 
            metrics_dict['NPV'] = npv[label]
            metrics_dict['Sensibility'] = sens[label] 
            metrics_dict['Specificity'] = spec[label]
            metrics_dict['Kappas'] = kappa[label] 

        final_dict[str(label)] = metrics_dict

    return final_dict

if __name__ == "__main__":

    json_file_name = 'other_metrics_2d_rgb_pixel'

    path_preds = r'Z:\grodriguez\CardiacOCT\preds_second_split\model_rgb_2d_preds'
    path_origs = r'Z:\grodriguez\CardiacOCT\data-2d\nnUNet_raw_data\Task602_CardiacOCT\labelsTs'
    num_classes = 13

    metrics_pixel = get_metrics_pixel()
    #metrics_detection = get_metrics_detection()

    with open("Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}.json".format(json_file_name), 'w') as f:
        json.dump(metrics_pixel, f, indent=4)



