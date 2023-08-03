import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import pandas as pd
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
sys.path.append("..") 
from utils.conversion_utils import get_prob_maps_list

def calculate_ece(true, pred, prob, num_bins=10):
    """Calculate the ECE given the true and predicted segmentations and the softmax values

    Args:
        true (np.array): true segmentation
        pred (np.array): predicted segmentation
        prob (np.array): softmax array
        num_bins (int, optional): number of bins in which to divide the probabilities. Defaults to 10.

    Returns:
        list, list, float: list of x and y for later plotting and the ECE
    """    
        
    #Get all true positives
    correct = (pred == true.astype(np.float32))

    #Define bins and divide probs in those bins
    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob, bins=b, right=True)

    o = 0
    x = []
    y = []
    for i in range(num_bins):
        #Get values correspondint to bin i
        mask = bins == i

        #Do ECE if there are values in bin
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob[mask]))

            accuracy = np.mean(correct[mask])
            conf = np.mean(prob[mask])
            y.append(accuracy)
            x.append(conf)

        ece = o / pred.shape[0]

    return x, y, ece

def reliability_diagram(x, y, ece):
    """Plot calibration curve

    Args:
        x (list): list with the accuracy for each bin
        y (list): list with confidence for each bin
        ece (float): ECE
    """    

    plt.plot(x, y)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal reference line
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibration Curve (ECE = {ece:.4f})')
    plt.show()



def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=str, default='Z:/grodriguez/CardiacOCT/preds_second_split/model_rgb_2d_preds')
    parser.add_argument('--label', type=int, default=4)
    parser.add_argument('--excel_name', type=str, default='ece_model_rgb_v2')
    args, _ = parser.parse_known_args(argv)

    orig_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task602_CardiacOCT/labelsTs'
    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')

    preds_list = os.listdir(args.preds_path)
    orig_list = os.listdir(orig_path)

    error_pd = pd.DataFrame(columns=['pullback', 'frame', 'type', 'ece'])

    for orig in orig_list:

        frame_name = orig.split('.')[0]
        files_for_frame = []

        error_list = []

        for pred in preds_list:
            
            if frame_name in pred:
                files_for_frame.append(pred)

        npz_file = [npz for npz in files_for_frame if npz.endswith('npz')][0]
        nifti_file = [nifti for nifti in files_for_frame if nifti.endswith('nii.gz')][0]

        orig_seg = sitk.ReadImage(os.path.join(orig_path, nifti_file))
        orig_seg_data = sitk.GetArrayFromImage(orig_seg)[0]

        pred_seg = sitk.ReadImage(os.path.join(args.preds_path, nifti_file))
        pred_seg_data = sitk.GetArrayFromImage(pred_seg)[0]
        prob_map = np.load(os.path.join(args.preds_path, npz_file))

        print('Checking ', nifti_file, npz_file)
        
        prob_img = get_prob_maps_list(prob_map)

        #Check again the case for the weird shapes
        if prob_img.shape[1] == 690:

            true_seg_crop = orig_seg_data[6:696, 6:697]
            pred_seg_crop = pred_seg_data[6:696, 6:697]

        else:
            true_seg_crop = orig_seg_data[6:697, 6:697]
            pred_seg_crop = pred_seg_data[6:697, 6:697]

        #Uniques for the labels
        labels_pred = np.unique(pred_seg_crop)
        labels_orig = np.unique(true_seg_crop)

        # So we want to compute the ECE for one label (lipid, for example). Maybe it does not make sense to compute it 
        # in FP or FN, but this code is for that. 

        if args.label in labels_pred and args.label not in labels_orig:
            cm_type = 'FP'

            #For FP, we take the pixels that contain the label we want in the predicted segmentation. We take those same pixels in the 
            #true segmentation
            true_label_crop = true_seg_crop[pred_seg_crop == args.label]
            pred_label_crop = pred_seg_crop[pred_seg_crop == args.label]
            prob_label_img = prob_img[args.label][pred_seg_crop == args.label]
            _, _, ece = calculate_ece(true_label_crop, pred_label_crop, prob_label_img)

        elif args.label in labels_orig and args.label not in labels_pred:
            cm_type = 'FN'
            
            #For FN, we take instead the pixels in the true segmentation that contain the labels, and we take those same pixels in the
            #predicted segmentation
            true_label_crop = true_seg_crop[true_seg_crop == args.label]
            pred_label_crop = pred_seg_crop[true_seg_crop == args.label]
            prob_label_img = prob_img[args.label][true_label_crop == args.label]
            _, _, ece = calculate_ece(true_label_crop, pred_label_crop, prob_label_img)

        #We dont calculate anything in TN
        elif args.label not in labels_orig and args.label not in labels_pred:
            cm_type = 'TN'
            ece = np.nan

        #We do the same as in FN
        else:
            cm_type = 'TP'
            true_label_crop = true_seg_crop[true_seg_crop == args.label]
            pred_label_crop = pred_seg_crop[true_seg_crop == args.label]
            prob_label_img = prob_img[args.label][true_seg_crop == args.label]
            _, _, ece = calculate_ece(true_label_crop, pred_label_crop, prob_label_img)

        print(cm_type)
        print('ECE:', ece, '\n')

        #Obtain format of pullback name (it's different than in the dataset counting)
        filename = nifti_file.split('_')[0]
        first_part = filename[:3]
        second_part = filename[3:-4]
        third_part = filename[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Obtain pullback name
        n_pullback = nifti_file.split('_')[1]
        pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

        #Obtain nº frame
        n_frame = nifti_file.split('_')[2][5:]
        
        error_list.append(pullback_name)
        error_list.append(n_frame)
        error_list.append(cm_type)
        error_list.append(ece)

        error_pd = error_pd.append(pd.Series(error_list, index=error_pd.columns[:len(error_list)]), ignore_index=True)

    error_pd.to_excel('Z:/grodriguez/CardiacOCT/info-files/uncertainty/{}.xlsx'.format(args.excel_name))


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)

    