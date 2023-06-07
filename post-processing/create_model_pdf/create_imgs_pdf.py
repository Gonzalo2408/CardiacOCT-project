import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
import argparse
import sys
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(1, '/mnt/netcache/diag/grodriguez/CardiacOCT/post-processing')
from output_handling import create_annotations_lipid, create_annotations_calcium

def detect_tcfa(fct, arc):

    if fct == 'nan':
        return  0
    
    if int(fct) < 65 and int(arc) >= 90:
        return 1
    
    else:
        return 0
    

def calcium_score(arc, thickness):

    score = 0

    if int(arc) > 180:
        score += 2

    if int(thickness) > 0.5:
        score += 1

    return score


def find_labels(seg):

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


def map_colors(segmentation):

    #Specify color map
    color_map = {
    0: (0, 0, 0),        #background
    1: (255, 0, 0),      #lumen
    2: (0, 255, 0),      #guide
    3: (0, 0, 255),      #initma
    4: (255, 255, 0),    #lipid
    5: (255, 255, 255),  #calcium
    6: (255, 0, 255),    #media
    7: (146, 0, 0),      #catheter
    8: (255, 123, 0),    #sidebranch
    9: (230, 141, 230),  #rt
    10: (0, 255, 255),   #wt
    11: (65, 135, 100),  #scad
    12: (208, 190, 161), #rupture
    }

    #Convert the labels array into a color-coded image
    h, w = segmentation.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_img[segmentation == label] = color

    return color_img



def resize_image(raw_frame, downsample = True):

    frame_image = sitk.GetImageFromArray(raw_frame)

    if downsample == True:
        new_shape = (704, 704)

    else:
        new_shape = (1024, 1024)


    new_spacing = (frame_image.GetSpacing()[0]*sitk.GetArrayFromImage(frame_image).shape[1]/new_shape[0],
                        frame_image.GetSpacing()[1]*sitk.GetArrayFromImage(frame_image).shape[1]/new_shape[0])

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(new_shape)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)

    resampled_seg = resampler.Execute(frame_image)
    resampled_seg_frame = sitk.GetArrayFromImage(resampled_seg)

    return resampled_seg_frame

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    mask = np.expand_dims(mask, 0)

    return mask


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/preds-test-set/predicted_results_model1_2d_updated')
    parser.add_argument('--pdf_name', type=str)
    args, _ = parser.parse_known_args(argv)
    
    raw_imgs_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans-DICOM'
    segs_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/segmentations-ORIGINALS'

    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

    test_set = annots[annots['Set'] == 'Testing']['Pullback'].tolist()

    pdf = PdfPages('/mnt/netcache/diag/grodriguez/CardiacOCT/post-processing/models_reports/{}.pdf'.format(args.pdf_name))

    for file in test_set:

        #Reading raw image
        print('Procesing case ', file)
        image = sitk.ReadImage(os.path.join(raw_imgs_path, file+'.dcm'))
        image_data = sitk.GetArrayFromImage(image)

        #Reading seg
        seg = sitk.ReadImage(os.path.join(segs_path, file+'.nii.gz'))
        seg_data = sitk.GetArrayFromImage(seg)

        #Getting name, id, pullback and frames variables
        frames_with_annot = annots.loc[annots['Pullback'] == file]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')] 
        n_pullback = annots.loc[annots['Pullback'] == file]['Nº pullback'].values[0]
        
        patient_name = annots.loc[annots['Pullback'] == file]['Patient'].values[0]
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])

        spacing = annots.loc[annots['Pullback'] == file]['Spacing'].values[0]

        for frame in frames_list:
            
            #Get raw frame and seg
            raw_img_to_plot = image_data[frame,:,:,:]
            seg_to_plot = seg_data[frame,:,:]

            #Check if resize is neeeded (shape should be (704, 704))
            if seg_to_plot.shape == (1024, 1024) and spacing == 0.006842619:
                resampled_seg_frame = resize_image(seg_to_plot)

            elif seg_to_plot.shape == (1024, 1024) and spacing == 0.009775171:
                resampled_seg_frame = seg_to_plot[160:864, 160:864]

            elif seg_to_plot.shape == (704, 704) and (spacing == 0.014224751 or spacing == 0.014935988):
                resampled_seg_frame = resize_image(seg_to_plot, False)
                resampled_seg_frame = resampled_seg_frame[160:864, 160:864]

            else:
                resampled_seg_frame = seg_to_plot

            #Apply mask to both seg and image
            circular_mask = create_circular_mask(resampled_seg_frame.shape[0], resampled_seg_frame.shape[1], radius=346)
            masked_resampled_frame = np.invert(circular_mask) * resampled_seg_frame

            #Get corresponding prediction for a given model (in args.preds)
            pred_seg_name = '{}_{}_frame{}_{}.nii.gz'.format(patient_name.replace('-', ''), n_pullback, frame, "%03d" % id)
            pred_seg = sitk.ReadImage(os.path.join(args.preds, pred_seg_name))
            pred_seg_data = sitk.GetArrayFromImage(pred_seg)

             #Lipid and calcium measurements (both preds and original seg)
            lipid_img_pred, _ , fct_pred, lipid_arc_pred, _ = create_annotations_lipid(pred_seg_data[0])
            calcium_img_pred, _ , _, cal_arc_pred, cal_thickness_pred, _ = create_annotations_calcium(pred_seg_data[0])

            _, _ , fct_orig, lipid_arc_orig, _ = create_annotations_lipid(masked_resampled_frame[0])
            _, _ , _, cal_arc_orig, cal_thickness_orig, _ = create_annotations_calcium(masked_resampled_frame[0])

            #Detect TCFA
            tcfa_pred = detect_tcfa(fct_pred, lipid_arc_pred)
            tcfa_orig = detect_tcfa(fct_orig, lipid_arc_orig)

            #Detect calcium score
            cal_score_pred = calcium_score(cal_arc_pred, cal_thickness_pred)
            cal_score_orig = calcium_score(cal_arc_orig, cal_thickness_orig)

            #Get values for table
            sb_orig, rt_orig, wt_orig, scad_orig, rupture_orig = find_labels(resampled_seg_frame)
            sb_pred, rt_pred, wt_pred, scad_pred, rupture_pred = find_labels(pred_seg_data[0])

            #Change segmentation colors
            final_orig_seg = map_colors(masked_resampled_frame[0])
            final_pred_seg = map_colors(pred_seg_data[0])

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
            #plt.subplots_adjust(wspace=0.1, hspace=0.1)
            #axes[5].get_position().set_points([0.1, 0.2, 0.9, 0.9])

            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)