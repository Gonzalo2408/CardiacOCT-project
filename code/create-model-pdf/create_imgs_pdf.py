import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
import argparse
import cv2
import sys
import skimage
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(1, '/mnt/netcache/diag/grodriguez/CardiacOCT/code/utils')
from postprocessing import create_annotations_lipid, create_annotations_calcium
from counts_utils import create_image_png

def detect_tcfa(fct, arc):
    """Checks if a frame contains a TCFA or not

    Args:
        fct (int): fibrous cap thickness
        arc (int): lipid arc

    Returns:
        int: either 0 or 1, indicating presence of TCFA
    """    

    if fct == 'nan':
        return  0
    
    if int(fct) < 65 and int(arc) >= 90:
        return 1
    
    else:
        return 0
    

def calcium_score(arc, thickness):
    """Obtain calcium score (ranging 0-4). Sadly, we cannot get the calcium length (yet)

    Args:
        arc (int): calcium arc
        thickness (int): calcium thickness

    Returns:
        int: calcium score
    """    

    score = 0

    if int(arc) > 180:
        score += 2

    if int(thickness) > 0.5:
        score += 1

    return score


def find_labels(seg):
    """Get binary values for each label in the segmentation

    Args:
        seg (np.array): image with segmentation

    Returns:
        int: sidebranch
        int: red thrombus
        int: white thrombus
        int: dissection
        int: plaque rupture

    """    

    labels = np.unique(seg)

    if 8 in labels: sb = 1
    else: sb = 0

    if 9 in labels: rt = 1 
    else: rt = 0

    if 10 in labels: wt = 1  
    else: wt = 0

    if 11 in labels: scad = 1
    else: scad = 0

    if 12 in labels: rupture = 1
    else: rupture = 0

    return sb, rt, wt, scad, rupture

def morph_operation(img):
    """Performs closing morphological operation (we only perfom this on certain labels)

    Args:
        img (np.array): array with segmentation

    Returns:
        np.array np.array: array with closing operation
    """    
    cols, rows = img.shape
    to_process = np.zeros((cols, rows))
    orig_split = np.zeros((cols, rows))

    for col in range(cols):
        for row in range(rows):

            #Do only morph operation in wall, lipid, calcium and intima
            if img[col, row] == 3:
                to_process[col, row] = 3

            elif img[col, row] == 4:
                to_process[col, row] = 4

            elif img[col, row] == 5:
                to_process[col, row] = 5

            elif img[col, row] == 6:
                to_process[col, row] = 6
                
            else:
                to_process[col, row] = 0
                orig_split[col, row] = img[col, row]

    kernel = skimage.morphology.disk(5)
    closing = cv2.morphologyEx(to_process, cv2.MORPH_CLOSE, kernel)

    #Merge processed regions with the remaining (which are in the original array)
    final = closing + orig_split

    return final


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task601_CardiacOCT/labelsTs')
    parser.add_argument('--preds', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/model7_preds')
    parser.add_argument('--pdf_name', type=str)
    parser.add_argument('--morph_op', type=bool, default=False)
    args, _ = parser.parse_known_args(argv)
    
    raw_imgs_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/DICOM'

    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')

    test_set = annots[annots['Set'] == 'Testing']['Pullback'].tolist()

    pdf = PdfPages('/mnt/netcache/diag/grodriguez/CardiacOCT/info-files/models_reports/{}.pdf'.format(args.pdf_name))

    if args.morph_op:
        print('Performing closing morphological operation')

    for file in test_set:

        #Reading raw image
        print('Procesing case ', file)
        image = sitk.ReadImage(os.path.join(raw_imgs_path, file+'.dcm'))
        image_data = sitk.GetArrayFromImage(image)

        #Getting name, id, pullback and frames variables
        frames_with_annot = annots.loc[annots['Pullback'] == file]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')] 
        n_pullback = annots.loc[annots['Pullback'] == file]['Nº pullback'].values[0]
        
        patient_name = annots.loc[annots['Pullback'] == file]['Patient'].values[0]
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])

        for frame in frames_list:

            #Get corresponding prediction for a given model (in args.preds)
            pred_seg_name = '{}_{}_frame{}_{}.nii.gz'.format(patient_name.replace('-', ''), n_pullback, frame, "%03d" % id)
            if not os.path.exists(os.path.join(args.preds, pred_seg_name)):
                continue
                
            pred_seg = sitk.ReadImage(os.path.join(args.preds, pred_seg_name))
            pred_seg_data = sitk.GetArrayFromImage(pred_seg)[0]

            #Perform or don't perform morphological closing
            if args.morph_op:
                pred_seg_data = morph_operation(pred_seg_data)

            #Reading orig seg
            seg = sitk.ReadImage(os.path.join(args.orig, pred_seg_name))
            seg_to_plot = sitk.GetArrayFromImage(seg)[0]
            
            #Get raw frame and seg
            raw_img_to_plot = image_data[frame,:,:,:]

            #Lipid and calcium measurements (both preds and original seg)
            lipid_img_pred, _ , fct_pred, lipid_arc_pred, _ = create_annotations_lipid(pred_seg_data)
            calcium_img_pred, _ , _, cal_arc_pred, cal_thickness_pred, _ = create_annotations_calcium(pred_seg_data)

            _, _ , fct_orig, lipid_arc_orig, _ = create_annotations_lipid(seg_to_plot)
            _, _ , _, cal_arc_orig, cal_thickness_orig, _ = create_annotations_calcium(seg_to_plot)

            #Detect TCFA
            tcfa_pred = detect_tcfa(fct_pred, lipid_arc_pred)
            tcfa_orig = detect_tcfa(fct_orig, lipid_arc_orig)

            #Detect calcium score
            cal_score_pred = calcium_score(cal_arc_pred, cal_thickness_pred)
            cal_score_orig = calcium_score(cal_arc_orig, cal_thickness_orig)

            #Get values for table
            sb_orig, rt_orig, wt_orig, scad_orig, rupture_orig = find_labels(seg_to_plot)
            sb_pred, rt_pred, wt_pred, scad_pred, rupture_pred = find_labels(pred_seg_data)

            #Change segmentation colors
            final_orig_seg = create_image_png(seg_to_plot)
            final_pred_seg = create_image_png(pred_seg_data)

            fig, axes = plt.subplots(2, 3, figsize=(50,50), constrained_layout = True)

            axes = axes.flatten()

            axes[0].set_title('Raw frame', fontsize=75)
            axes[0].imshow(raw_img_to_plot)
            axes[0].get_xaxis().set_visible(False)
            axes[0].get_yaxis().set_visible(False)
            
            axes[1].set_title('Raw segmentation', fontsize=75)
            axes[1].imshow(final_orig_seg)
            axes[1].get_xaxis().set_visible(False)
            axes[1].get_yaxis().set_visible(False)
            
            axes[2].set_title('Pred segmentation', fontsize=75)
            axes[2].imshow(final_pred_seg)
            axes[2].get_xaxis().set_visible(False)
            axes[2].get_yaxis().set_visible(False)

            axes[3].set_title('Lipid measures', fontsize=75)
            axes[3].imshow(final_pred_seg, alpha=0.5)
            axes[3].imshow(lipid_img_pred, alpha=0.8)
            axes[3].get_xaxis().set_visible(False)
            axes[3].get_yaxis().set_visible(False)

            axes[4].set_title('Calcium measures', fontsize=75)
            axes[4].imshow(final_pred_seg, alpha=0.5)
            axes[4].imshow(calcium_img_pred ,alpha=0.8)
            axes[4].get_xaxis().set_visible(False)
            axes[4].get_yaxis().set_visible(False)

            fig.suptitle('Pullback: {}. Frame {}'.format(file, frame), fontsize=75) 

            #Table with measures and more data
            columns = ('Raw', 'Predicted')
            rows = [x for x in ('TCFA', 'Calcium score', 'Sidebranch', 'Rthrombus', 'Wthrombus', 'Dissection', 'Rupture')]
            data = [[tcfa_orig, tcfa_pred],
                    [cal_score_orig, cal_score_pred],
                    [sb_orig, sb_pred],
                    [rt_orig, rt_pred],
                    [wt_orig, wt_pred],
                    [scad_orig, scad_pred],
                    [rupture_orig, rupture_pred]]
            
            n_rows = len(data)
            cell_text = []

            for row in range(n_rows):
                y_offset = data[row]
                cell_text.append([x for x in y_offset])

            #Add a table at the bottom of the axes
            table = plt.table(cellText=cell_text,
                                rowLabels=rows,
                                colLabels=columns,
                                loc='center')
            
            axes[5].axis('off')
            table.scale(0.5, 9)
            table.set_fontsize(75)

            pdf.savefig(fig)
            plt.close(fig)
    
    pdf.close()

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)