import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
import argparse
import cv2
import sys
import skimage
from typing import Tuple
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(1, '/mnt/netcache/diag/grodriguez/CardiacOCT/code/utils')
from postprocessing import create_annotations_lipid, create_annotations_calcium
from counts_utils import create_image_png


class PDF_report:

    def __init__(self, path_preds: str, path_origs: str, morph_op: bool):

        self.path_preds = path_preds
        self.path_origs = path_origs
        self.morph_op = morph_op

    def read_img(self, path):

        img_data = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img_data)[0]

        return img

    def detect_tcfa(self, fct: float, arc: float) -> int:
        """Checks if a frame contains a TCFA or not

        Args:
            fct (float): fibrous cap thickness
            arc (float): lipid arc

        Returns:
            int: either 0 or 1, indicating presence of TCFA
        """

        if fct == 'nan':
            return 0

        if int(fct) < 65 and int(arc) >= 90:
            return 1

        else:
            return 0

    def calcium_score(self, arc: float, thickness: float) -> int:
        """Obtain calcium score (ranging 0-4). Sadly, we cannot get the calcium length (yet)

        Args:
            arc (float): calcium arc
            thickness (float): calcium thickness

        Returns:
            int: calcium score
        """

        score = 0

        if int(arc) > 180:
            score += 2

        if int(thickness) > 0.5:
            score += 1

        return score

    def find_labels(self, seg: np.array) -> Tuple[int, int, int, int, int]:
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

        if 8 in labels:
            sb = 1
        else:
            sb = 0

        if 9 in labels:
            rt = 1
        else:
            rt = 0

        if 10 in labels:
            wt = 1
        else:
            wt = 0

        if 11 in labels:
            scad = 1
        else:
            scad = 0

        if 12 in labels:
            rupture = 1
        else:
            rupture = 0

        return sb, rt, wt, scad, rupture

    def morph_operation(self, img: np.array) -> np.array:
        """Performs closing morphological operation (we only perfom this on certain labels)

        Args:
            img (np.array): array with segmentation

        Returns:
            np.array: array with closing operation
        """
        cols, rows = img.shape
        to_process = np.zeros((cols, rows))
        orig_split = np.zeros((cols, rows))

        for col in range(cols):
            for row in range(rows):

                # Do only morph operation in wall, lipid, calcium and intima
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

        # Merge processed regions with the remaining (which are in the original array)
        final = closing + orig_split

        return final

    def get_models_overview(self, raw_img: np.array, seg_name: str) -> matplotlib.figure.Figure:
        """Obtain the report to see all the segmentations of every model for every frame

        Args:
            raw_img (np.array): raw DICOM frame
            seg_name (str): name of the file containing the segmentation

        Returns:
            matplotlib.figure.Figure: figure with subplots containing the segmentations and raw image
        """

        dict_preds = {}

        orig_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path_origs, seg_name)))[0]
        final_orig_seg = create_image_png(orig_seg)

        # Iterate through every prediction folder
        for pred_folder in os.listdir(self.path_preds):

            pred_seg_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path_preds,
                                                                               pred_folder,
                                                                               seg_name)))[0]
            final_pred_seg = create_image_png(pred_seg_data)

            # Get specific substring for the plot title
            sub_pred_folder = pred_folder.split('_')
            dict_preds['_'.join(sub_pred_folder[:4])] = final_pred_seg

        fig, axes = plt.subplots(2, 3, figsize=(50, 50), constrained_layout=True)

        axes = axes.flatten()

        dict_preds = list(dict_preds.items())

        axes[0].set_title('Raw frame', fontsize=75)
        axes[0].imshow(raw_img)
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)

        axes[1].set_title('Raw segmentation', fontsize=75)
        axes[1].imshow(final_orig_seg)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)

        axes[3].set_title('Pseudo 3D k=1', fontsize=75)
        axes[3].imshow(dict_preds[0][1])
        axes[3].get_xaxis().set_visible(False)
        axes[3].get_yaxis().set_visible(False)

        axes[4].set_title('Pseudo 3D k=2', fontsize=75)
        axes[4].imshow(dict_preds[1][1])
        axes[4].get_xaxis().set_visible(False)
        axes[4].get_yaxis().set_visible(False)

        axes[5].set_title('Pseudo 3D k=3', fontsize=75)
        axes[5].imshow(dict_preds[2][1])
        axes[5].get_xaxis().set_visible(False)
        axes[5].get_yaxis().set_visible(False)

        axes[2].set_title('2D', fontsize=75)
        axes[2].imshow(dict_preds[3][1])
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)

        return fig

    def get_model_summary(self, raw_img: np.array, seg_name: str) -> matplotlib.figure.Figure:
        """Obtain standard model report (with measurements and table)

        Args:
            raw_img (np.array): Raw DICOM frame
            seg_name (str): name of the file containing the segmentation

        Returns:
            matplotlib.figure.Figure: figure containing the raw frame, manual and predicted segmentations,
            measurements and table with information
        """

        pred_seg = sitk.ReadImage(os.path.join(self.path_preds, seg_name))
        pred_seg_data = sitk.GetArrayFromImage(pred_seg)[0]

        # Perform or don't perform morphological closing
        if self.morph_op:
            pred_seg_data = self.morph_operation(pred_seg_data)

        # Reading orig seg
        seg_to_plot = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path_origs, seg_name)))[0]

        # Lipid and calcium measurements (both preds and original seg)
        lipid_img_pred, _, fct_pred, lipid_arc_pred, _ = create_annotations_lipid(pred_seg_data)
        calcium_img_pred, _, _, cal_arc_pred, cal_thickness_pred, _ = create_annotations_calcium(pred_seg_data)

        _, _, fct_orig, lipid_arc_orig, _ = create_annotations_lipid(seg_to_plot)
        _, _, _, cal_arc_orig, cal_thickness_orig, _ = create_annotations_calcium(seg_to_plot)

        # Detect TCFA
        tcfa_pred = self.detect_tcfa(fct_pred, lipid_arc_pred)
        tcfa_orig = self.detect_tcfa(fct_orig, lipid_arc_orig)

        # Detect calcium score
        cal_score_pred = self.calcium_score(cal_arc_pred, cal_thickness_pred)
        cal_score_orig = self.calcium_score(cal_arc_orig, cal_thickness_orig)

        # Get values for table
        sb_orig, rt_orig, wt_orig, scad_orig, rupture_orig = self.find_labels(seg_to_plot)
        sb_pred, rt_pred, wt_pred, scad_pred, rupture_pred = self.find_labels(pred_seg_data)

        # Change segmentation colors
        final_orig_seg = create_image_png(seg_to_plot)
        final_pred_seg = create_image_png(pred_seg_data)

        fig, axes = plt.subplots(2, 3, figsize=(50, 50), constrained_layout=True)

        axes = axes.flatten()

        axes[0].set_title('Raw frame', fontsize=75)
        axes[0].imshow(raw_img)
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
        axes[4].imshow(calcium_img_pred, alpha=0.8)
        axes[4].get_xaxis().set_visible(False)
        axes[4].get_yaxis().set_visible(False)

        # Table with measures and more data
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

        # Add a table at the bottom of the axes
        table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')

        axes[5].axis('off')
        table.scale(0.5, 9)
        table.set_fontsize(75)

        return fig


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', type=str, help='Path to the original segmentation files (in labelsTs folder)')
    parser.add_argument('--preds', type=str, help='Path to the predictions folder by the nnUNet')
    parser.add_argument('--pdf_name', type=str, help='Name of the PDF file without the .pdf')
    parser.add_argument('--morph_op', action='store_true', help='Perform morphological operation?')
    parser.add_argument('--overview', action='store_true', help="""Get overview of all models instead of for one model
                                                                (--preds argument can be empty in this case)""")
    args, _ = parser.parse_known_args(argv)

    raw_imgs_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/DICOM'

    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/info_files/train_test_split_final_v2.xlsx')

    test_set = annots[annots['Set'] == 'Testing']['Pullback'].tolist()

    if args.morph_op:
        print('Performing closing morphological operation')

    else:
        print('No morphological operations being applied')

    if args.overview:
        print('Getting overview of every training...')

    else:
        print('Getting report for model stored in ', args.preds)

    get_pdf = PDF_report(args.preds, args.orig, args.morph_op)

    for file in test_set:

        # Reading raw image
        print('Procesing case ', file)
        image = sitk.ReadImage(os.path.join(raw_imgs_path, file+'.dcm'))
        image_data = sitk.GetArrayFromImage(image)

        # Getting name, id, pullback and frames variables
        frames_with_annot = annots.loc[annots['Pullback'] == file]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        n_pullback = annots.loc[annots['Pullback'] == file]['Nº pullback'].values[0]

        patient_name = annots.loc[annots['Pullback'] == file]['Patient'].values[0]
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])

        pdf = PdfPages('/mnt/netcache/diag/grodriguez/CardiacOCT/info_files/models_reports/{}_{}.pdf'.format(file, args.pdf_name))

        for frame in frames_list:

            # Get corresponding prediction for a given model (in args.preds)
            seg_name = '{}_{}_frame{}_{}.nii.gz'.format(patient_name.replace('-', ''), n_pullback, frame, "%03d" % id)

            # Get raw frame and seg
            raw_img_to_plot = image_data[frame, :, :, :]

            if args.overview:
                fig = get_pdf.get_models_overview(raw_img_to_plot, seg_name)

            else:
                fig = get_pdf.get_model_summary(raw_img_to_plot, seg_name)

            fig.suptitle('Pullback: {}. Frame {}'.format(patient_name, frame), fontsize=75)
            pdf.savefig(fig)
            plt.close(fig)

        pdf.close()


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)
