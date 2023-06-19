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


def get_prob_maps_list(prob_map):

    #Takes the npz file with the prob maps and creates an array of shape (num_classes, img_shape)
    num_classes = 13
    probs_list = []

    for i in prob_map.items():

        for label in range(num_classes):
            probs_list.append(i[1][label][0])

    prob_img = np.zeros((num_classes, 691, 691))
    _, rows, cols = prob_img.shape

    for i in range(rows):
        for j in range(cols):

            prob_img[:, i, j] = np.array([probs_list[label][i, j] for label in range(num_classes)])

    return prob_img

############ Code taken from https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
def calculate_ece(y_true, y_pred, num_bins=10):
        
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true.astype(np.float32))
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    ece = o / y_pred.shape[0]

    x = prob_y[mask]
    y = correct[mask]

    return x, y, ece

def calculate_sce(y_true, y_pred, num_bins=10):
    
    classes = y_pred.shape[-1]

    o = 0
    for cur_class in range(classes):
        correct = (cur_class == y_true).astype(np.float32)
        prob_y = y_pred[..., cur_class]

        b = np.linspace(start=0, stop=1.0, num=num_bins)
        bins = np.digitize(prob_y, bins=b, right=True)

        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    sce = o / (y_pred.shape[0] * classes)

    x = prob_y[mask]
    y = correct[mask]

    return x, y, sce


def reliability_diagram(x, y):

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(x, y, "s-", label="CNN")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Reliability diagram / calibration curve")

    plt.tight_layout()
    plt.show()

#######################################################################################

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=str, default='Z:/grodriguez/CardiacOCT/preds-test-set/model7_preds')
    args, _ = parser.parse_known_args(argv)

    orig_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task512_CardiacOCT/labelsTs'
    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final.xlsx')

    preds_list = os.listdir(args.preds_path)
    orig_list = os.listdir(orig_path)

    num_classes = 13
    error_pd = pd.DataFrame(columns=['pullback', 'frame', 'ece', 'sce'])

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
        prob_map = np.load(os.path.join(args.preds_path, npz_file))

        print('Checking ', nifti_file, npz_file)
        
        prob_img = get_prob_maps_list(prob_map)

        true_seg_crop = orig_seg_data[6:697, 6:697]

        _, _, ece = calculate_ece(true_seg_crop.reshape(-1), prob_img.reshape(prob_img.shape[1]**2, num_classes))
        _, _, sce = calculate_sce(true_seg_crop.reshape(-1), prob_img.reshape(prob_img.shape[1]**2, num_classes))

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
        error_list.append(ece)
        error_list.append(sce)

        #reliability_diagram(x, y)

        error_pd = error_pd.append(pd.Series(error_list, index=error_pd.columns[:len(error_list)]), ignore_index=True)

    error_pd.to_excel('Z:/grodriguez/CardiacOCT/info-files/uncertainty/calibration_errors.xlsx')


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)

    