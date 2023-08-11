import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import pandas as pd
import sys
import argparse
from typing import Tuple
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class ECE_Conf:

    def __init__(self, true: np.array, pred: np.array, prob_img: np.array, num_bins: int):
        """Obtain reliability curve, ECE and confidence on lipid and calcium per frame

        Args:
            true (np.array): original segmentation (need to be cropped to the probability map size)
            pred (np.array): predicted segmentation (need to be cropped to the probability map size)
            prob_img (np.array): probability map with the softmax values per pixel
            num_bins (int): nº of bins to compute the ECE and plot the reliability curve
        """        

        self.true = true
        self.pred = pred
        self.prob_img = prob_img
        self.num_bins = num_bins

    def calculate_ece(self, total_true, total_pred, total_prob) -> Tuple[list, list, float]:
        """Calculate the ECE given the true and predicted segmentations and the softmax values

        Args:
            total_true (_type_): total ammount of pixels with original segmentation in the test set
            total_pred (_type_): total ammount of pixels with predicted segmentation in the test set
            total_prob (_type_): total ammount of pixels with the associated probability in the test set

        Returns:
            Tuple[list, list, float: list of x (confidences) and y (accuracies) for each bin plus the ECE
        """    
            
        #Get all true positives
        correct = total_pred == total_true

        #Define bins and divide probs in those bins
        b = np.linspace(start=0, stop=1.0, num=self.num_bins)
        bins = np.digitize(total_prob, bins=b, right=True)

        ece = 0
        x = []
        y = []
        for i in range(self.num_bins):

            #Get values corresponding to bin i
            mask = bins == i

            #Do ECE if there are values in that bin
            if np.any(mask):

                acc = np.mean(correct[mask])
                conf = np.mean(total_prob[mask])
                x.append(conf)
                y.append(acc)
                ece += (np.sum(mask) / len(total_pred)) * np.abs(acc - conf)

        return x, y, ece

    def reliability_diagram(self, x, y, ece, path_curves: str, name: str):
        """Plot calibration curve

        Args:
            path_curves (str): path to the folder in which to save the reliability curves in PNG
            name (str): name of the PNG file (it'll be just the frame name)
        """        
 

        plt.plot(x, y, '-o')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal reference line
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration Curve (ECE = {ece:.4f})')
        plt.savefig(os.path.join(path_curves, './{}.png'.format(name)))
        plt.close()

    def get_confidence(self, label: int) -> Tuple[str, float]: 
        """Obtain the average confidence for a given frame and a specific label (we are doing just lipid and calcium)

        Args:
            label (int): label to get the confidence

        Returns:
            Tuple[str, float]: str value either being TP, TN, FP or FN, and the average confidence on that frame
        """        

        #Uniques for the labels
        labels_pred = np.unique(self.pred)
        labels_orig = np.unique(self.true)

        #For FP, we take the pixels that contain the label we want in the predicted segmentation
        if label in labels_pred and label not in labels_orig:
            cm_type = 'FP'
            prob_label_img = self.prob_img[label][self.pred == label]
            conf = np.mean(prob_label_img)

        #For FN, we take the pixels in the predicted segmentation that correspond to the label in the true segmentation
        elif label in labels_orig and label not in labels_pred:
            cm_type = 'FN'
            prob_label_img = np.max(self.prob_img, axis=0)[self.true == label]
            conf = np.mean(prob_label_img)

        #We dont calculate anything in TN
        elif label not in labels_orig and label not in labels_pred:
            cm_type = 'TN'
            conf = np.nan

        #We do the same as in FP
        else:
            cm_type = 'TP'
            prob_label_img = self.prob_img[label][self.pred == label]
            conf = np.mean(prob_label_img)

        return cm_type, conf


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=str, default='Z:/grodriguez/CardiacOCT/preds_second_split/model_pseudo3d_1_preds')
    parser.add_argument('--orig_path', type=str, default='Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task604_CardiacOCT/labelsTs')
    parser.add_argument('--data_info', type=str, default='Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')
    parser.add_argument('--excel_name', type=str, default='ece_conf_pseudo3d_1')
    parser.add_argument('--path_curves', type=str, default='Z:/grodriguez/CardiacOCT/info-files/uncertainty/Pseudo 3D 1')
    args, _ = parser.parse_known_args(argv)

    annots = pd.read_excel(args.data_info)

    orig_list = os.listdir(args.orig_path)

    error_pd = pd.DataFrame(columns=['pullback', 'frame', 'type lipid', 'type calcium', 'conf lipid', 'conf_calcium'])

    #This is to compute the total ECE in the test set + total reliability curve
    total_orig = []
    total_pred = []
    total_probs = []

    for orig in orig_list:

        #List in which we append the stuff that will go to the dataframe
        error_list = []
        
        frame_name = orig.split('.')[0]

        #Get segmentation and probability map names
        nifti_file = '{}.nii.gz'.format(frame_name)
        npz_file = '{}.npz'.format(frame_name)

        orig_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.orig_path, nifti_file)))[0]
        pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.preds_path, nifti_file)))[0]
        prob_map = np.load(os.path.join(args.preds_path, npz_file))
        prob_img = np.transpose(list(prob_map.items())[0][1], (1,0,2,3))[0]
        prob_img_max = np.max(prob_img, axis=0)

        #Weird prboability maps with one missing pixel
        if prob_img.shape[1] == 690:
            orig_seg = orig_seg[6:696, 6:697]
            pred_seg = pred_seg[6:696, 6:697]

        else:
            orig_seg = orig_seg[6:697, 6:697]
            pred_seg = pred_seg[6:697, 6:697]

        #Could be better
        for i in range(orig_seg.shape[0]):
            for j in range(orig_seg.shape[1]):

                total_orig.append(orig_seg[i, j])
                total_pred.append(pred_seg[i, j])
                total_probs.append(prob_img_max[i, j])

        print('Checking ', '{}.nii.gz'.format(frame_name), '{}.npz'.format(frame_name))

        get_ece_conf = ECE_Conf(orig_seg, pred_seg, prob_img, 10)
        
        #Get confidences for lipid and calcium
        cm_type_lipid, conf_lipid = get_ece_conf.get_confidence(4)
        cm_type_cal, conf_cal = get_ece_conf.get_confidence(5)

        print('Lipid: {}. Calcium: {}'.format(cm_type_lipid, cm_type_cal))
        print('Conf lipid: {}. Conf calcium: {} \n'.format(conf_lipid, conf_cal))

        # Obtain format of pullback name (it's different than in the dataset counting)
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
        error_list.append(cm_type_lipid)
        error_list.append(cm_type_cal)
        error_list.append(conf_lipid)
        error_list.append(conf_cal)

        error_pd = error_pd.append(pd.Series(error_list, index=error_pd.columns[:len(error_list)]), ignore_index=True)

    #Calculate ECE
    x, y, ece = get_ece_conf.calculate_ece(np.asarray(total_orig), np.asarray(total_pred), np.asarray(total_probs))

    #Get reliability curve
    get_ece_conf.reliability_diagram(x, y, ece, args.path_curves, args.excel_name)

    error_pd.to_excel('Z:/grodriguez/CardiacOCT/info-files/uncertainty/{}.xlsx'.format(args.excel_name))


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)

    