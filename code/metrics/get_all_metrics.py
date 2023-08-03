import SimpleITK as sitk
import os
import json
import numpy as np
import pandas as pd
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
import argparse
sys.path.append("..") 
from utils.counts_utils import merge_frames_into_pullbacks
from utils.postprocessing import create_annotations_lipid, create_annotations_calcium, compute_arc_dices
from utils.metrics_utils import mean_metrics, calculate_confusion_matrix, metrics_from_cm


class Metrics:

    def __init__(self, orig_folder: str, preds_folder_name: str, preds_folder:str, data_info:str, model_id:str):
        """Class to get all the metrics files for a specific model

        Args:
            orig_folder (str): path to your original labels folder (just the raw_data -> TaskXXX -> labelsTs)
            preds_folder_name (str): specific name of the folder that contains your model predictions in the preds folder
            preds_folder (str): path to your preds folder
            data_info (str): path to the Excel file with the patients data (i.e train_test_split_v2.xlsx)
            model_id (str): arbitrary ID for a specific model (note that this will be used for naming all the generated metrics files!)
        """        

        self.orig_folder = orig_folder
        self.preds_folder_name = preds_folder_name
        self.preds_folder = preds_folder
        self.data_info = pd.read_excel(data_info)
        self.model_id = model_id     
        self.num_classes = 13

    def get_patient_data(self, file: str) -> str:
        """Processes the name of the file so we retrieve the pullback name

        Args:
            file (str): raw filename of prediction

        Returns:
            str: pullback name processed
        """        

        #Get patient name
        patient_name_raw = file.split('_')[0]
        first_part = patient_name_raw[:3]
        second_part = patient_name_raw[3:-4]
        third_part = patient_name_raw[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Get pullback_name
        n_pullback = file.split('_')[1]
        pullback_name = self.data_info[(self.data_info['Nº pullback'] == int(n_pullback)) & (self.data_info['Patient'] == patient_name)]['Pullback'].values[0]

        return pullback_name

    def dice_per_frame(self):
        """Obtain the DICE per frame, which are stores in a JSON file
        """        

        print('Getting DICE per frame...')
        json_results_file = os.path.join(self.preds_folder, self.preds_folder_name, 'summary.json')

        #Load summary file generated by nnUnet
        with open(json_results_file) as f:
            summary = json.load(f)

        final_dict = {}

        for file in os.listdir(os.path.join(self.preds_folder, self.preds_folder_name)):

            if file.endswith('.nii.gz'):

                list_dicts_per_frame = []

                #Get pullback name
                pullback_name = self.get_patient_data(file)
                frame = file.split('_')[2][5:]

                #Get DICE score from frame by looking at the json file
                for sub_dict in summary['results']['all']:
                    
                    if sub_dict['test'] == '/mnt/netcache/diag/grodriguez/CardiacOCT/preds_second_split/{}/{}'.format(self.preds_folder_name, file):
                        list_dicts_per_frame.append({k: v for i, (k, v) in enumerate(sub_dict.items()) if i < len(sub_dict) - 2})
                        break
                    else:
                        continue

                #Include frame
                mean_result = mean_metrics(list_dicts_per_frame)
                mean_result['Frame'] = frame
                mean_result['Pullback'] = pullback_name

                final_dict[file] = mean_result

        #Write final dict in a json file
        with open("Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_dice_frame.json".format(self.model_id), 'w') as f:
            json.dump(final_dict, f, indent=4)

        print('Done! Find your DICE results at Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_dice_frame.json'.format(self.model_id))
        print('############################\n')


    def dice_per_pullback(self):
        """Same as before, but for pullback level
        """        
        print('Getting DICE per pullback...')

        #Get frames that belong to each pullback
        merged_pullbacks = merge_frames_into_pullbacks(os.path.join(self.preds_folder, self.preds_folder_name))

        final_dict = {}

        for pullback in merged_pullbacks.keys():

            pullback_name = self.get_patient_data(pullback)

            print('Pullback ', pullback)

            dices_dict = {}

            cm_total = np.zeros((self.num_classes, self.num_classes), dtype=np.int)

            for frame in merged_pullbacks[pullback]:

                #Load original and pred segmentation
                seg_map_data_pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.preds_folder, self.preds_folder_name, frame)))[0]
                seg_map_data_orig = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.orig_folder, frame)))[0]

                #Sum cm for each frame so at the end we get the CM for the whole pullback
                cm = calculate_confusion_matrix(seg_map_data_orig, seg_map_data_pred, range(self.num_classes))       
                cm_total += cm

            dice, _, _, _, _, _ = metrics_from_cm(cm_total)

            #Create dictionary with the labels and DICEs
            for label in range(1,len(dice)):

                #Check for nans and put them in string-like
                if math.isnan(dice[label]):
                    dices_dict[str(label)] = "NaN"

                else:

                    dices_dict[str(label)] = dice[label] 

            final_dict[pullback_name] = dices_dict

        with open("Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_dice_pullback.json".format(self.model_id), 'w') as f:
            json.dump(final_dict, f, indent=4)


        print('Done! Find your DICE results at Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_dice_pullback.json'.format(self.model_id))
        print('############################\n')

    def get_other_metrics_detection(self):
        """Obtain PPV, NPV, sensitivity, specificity and cohen's kappa based on labels that appear in each frame
        """    

        print('Getting other metrics for detection...')
        cm_dict = {}

        for file in os.listdir(os.path.join(self.preds_folder, self.preds_folder_name)):

            if file.endswith('nii.gz'):
            
                print('Checking case', file)
                orig_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.orig_folder, file)))
                pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.preds_folder, self.preds_folder_name, file)))

                #Get labels that apppear in frame
                unique_orig = np.unique(orig_seg)
                unique_pred = np.unique(pred_seg)

                #First, we want to check which labels occur and do not occur in both prediction and original
                for label in range(1, self.num_classes):

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

                    #Create dictionary with the CM values for every label
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

        with open("Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_other_metrics_detection.json".format(self.model_id), 'w') as f:
            json.dump(final_dict, f, indent=4)

        print('Done! Find your other metrics (detection) results at Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_other_metrics_detection.json'.format(self.model_id))
        print('############################\n')

        return final_dict
    
    def get_other_metrics_pixel(self):
        """Obtain PPV, NPV, sensitivity, specificity and cohen's kappa based on pixel level
        """    

        print('Getting other metrics for pixel...')
        cm_total = np.zeros((self.num_classes, self.num_classes))

        for file in os.listdir(os.path.join(self.preds_folder, self.preds_folder_name)):

            if file.endswith('nii.gz'):
            
                print('Checking case', file)
                orig_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.orig_folder, file)))
                pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.preds_folder, self.preds_folder_name, file)))

                #In this case, we look at every pixel in the image (we can do same as in the get_dice_pullback_level)
                cm = calculate_confusion_matrix(orig_seg, pred_seg, range(self.num_classes))       
                cm_total += cm

        _, ppv, npv, sens, spec, kappa = metrics_from_cm(cm_total)
        final_dict = {}

        #Create dictionary with the labels and metrics (except DICE)
        for label in range(1, len(ppv)):

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

        with open("Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_other_metrics_pixel.json".format(self.model_id), 'w') as f:
            json.dump(final_dict, f, indent=4)

        print('Done! Find your other metrics (pixel) results at Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_other_metrics_pixel.json'.format(self.model_id))
        print('############################\n')
        

    def get_arc_dice_per_frame(self):
        """Obtain the DICE score for lipid and calcium arcs per frame. The results are saved in an Excel file
        """        

        print('Getting lipid and calcium arc DICE per frame...')

        orig_segs = os.listdir(self.orig_folder)

        new_excel_data = pd.DataFrame(columns=['pullback', 'frame', 'DICE lipid', 'DICE calcium'])

        for seg in orig_segs:

            list_data = []

            print('Case ', seg)

            #Obtain format of pullback name as in the beginning
            pullback_name = self.get_patient_data(seg)

            #Obtain nº frame and set (train/test)
            n_frame = seg.split('_')[2][5:]

            #Read original and pred segmentation
            orig_img = sitk.ReadImage(os.path.join(self.orig_folder, seg))
            orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

            pred_img = sitk.ReadImage(os.path.join(self.preds_folder, self.preds_folder_name, seg))
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

        new_excel_data.to_excel('Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_arc_dices_per_frame.xlsx'.format(self.model_id))

        print('Done! Find your arc DICE results per frame at Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_arc_dices_per_frame.xlsx'.format(self.model_id))
        print('############################\n')


    def get_arc_dice_per_pullback(self):
        """Obtain the DICE score for lipid arc and calcium arc on pullback level (i.e confusion matrix is computed for the whole pullback)
        """      

        print('Getting lipid and calcium arc DICE per pullback...')
        pullback_dict = merge_frames_into_pullbacks(os.path.join(self.preds_folder, self.preds_folder_name))
        new_excel_data = pd.DataFrame(columns=['pullback', 'DICE lipid', 'DICE calcium'])

        for pullback in pullback_dict.keys():

            print('Pullback ', pullback)

            #In order to get DICEs pullback-level, we obtain all of the bin IDs for lipid in every frame with annotation in a pullback
            list_data = []

            #Obtain pullback name
            pullback_name = self.get_patient_data(pullback)

            tp_total_lipid = 0
            fp_total_lipid = 0
            fn_total_lipid = 0

            tp_total_calcium = 0
            fp_total_calcium = 0
            fn_total_calcium = 0

            
            for frame in pullback_dict[pullback]:

                print('Checking frame ', frame)
                
                orig_img = sitk.ReadImage(os.path.join(self.orig_folder, frame))
                orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

                pred_img = sitk.ReadImage(os.path.join(self.preds_folder, self.preds_folder_name, frame))
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

            list_data.append(pullback_name)
            list_data.append(dice_score_pullback_lipid)
            list_data.append(dice_score_pullback_calcium)

            new_excel_data = new_excel_data.append(pd.Series(list_data, index=new_excel_data.columns[:len(list_data)]), ignore_index=True)

        new_excel_data.to_excel('Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_arc_dices_per_pullback.xlsx'.format(self.model_id))

        print('Done! Find your arc DICE results per pullback at Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}_arc_dices_per_pullback.xlsx'.format(self.model_id))
        print('############################\n')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_folder', type=str, default='Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task602_CardiacOCT/labelsTs')
    parser.add_argument('--preds_folder_name', type=str)
    parser.add_argument('--preds_folder', type=str, default='Z:/grodriguez/CardiacOCT/preds_second_split')
    parser.add_argument('--data_info', type=str, default='Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')
    parser.add_argument('--model_id')
    args, _ = parser.parse_known_args(argv)

    args = parser.parse_args()

    metrics = Metrics(args.orig_folder, args.preds_folder_name, args.preds_folder, args.data_info, args.model_id)

    #Call all metrics functions (if you want to do a specific, just comment the others out)
    metrics.dice_per_frame()
    metrics.dice_per_pullback()
    metrics.get_other_metrics_detection()
    metrics.get_arc_dice_per_frame()
    metrics.get_arc_dice_per_pullback()

if __name__ == "__main__":
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)