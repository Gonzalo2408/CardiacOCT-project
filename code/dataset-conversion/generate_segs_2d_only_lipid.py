import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import argparse
import sys
sys.path.append("..") 
from utils.conversion_utils import create_circular_mask, resize_image, check_uniques

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Z:/grodriguez/CardiacOCT/data-original/segmentations-ORIGINALS')
    parser.add_argument('--task', type=str, default='Task512_CardiacOCT')
    parser.add_argument('--do_only_lipid', type=bool, default=False)
    args, _ = parser.parse_known_args(argv)

    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final.xlsx')

    for filename in os.listdir(args.data):

        #Geting patient ID from pullback
        patient_name = "-".join(filename.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        #Output folder
        if belonging_set == 'Testing':
            new_path_segs = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/labelsTs'.format(args.task)

        else:
            new_path_segs = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/labelsTr'.format(args.task)


        pullback_name = filename.split('.')[0]
        print('Checking ', pullback_name)

        #Get ID and nº of pullback
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

        #Read segmentation file
        orig_seg = sitk.ReadImage(args.data + '/' + filename)
        orig_seg_pixel_array = sitk.GetArrayFromImage(orig_seg)

        # Find the frames with annotations
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

        for frame in range(len(orig_seg_pixel_array)):

            if frame in frames_list:

                final_path = new_path_segs + '/' + patient_name.replace('-', '') + '_{}_frame{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id)

                #Check that a seg has already been generated
                if os.path.exists(final_path):
                    print('File already exists. Skip')
                    continue

                raw_frame = orig_seg_pixel_array[frame,:,:]
                spacing = annots.loc[annots['Pullback'] == pullback_name]['Spacing'].values[0]

                #Get only frames that have lipid
                if args.do_only_lipid:
                    if 4 not in np.unique(raw_frame):
                        continue

                #Check if resize is neeeded (shape should be (704, 704) and with the correct spacing)
                if raw_frame.shape == (1024, 1024) and spacing == 0.006842619:
                    resampled_seg_frame = resize_image(raw_frame)

                elif raw_frame.shape == (1024, 1024) and spacing == 0.009775171:
                    resampled_seg_frame = raw_frame[160:864, 160:864]

                elif raw_frame.shape == (704, 704) and (spacing == 0.014224751 or spacing == 0.014935988):
                    resampled_seg_frame = resize_image(raw_frame, False)
                    resampled_seg_frame = resampled_seg_frame[160:864, 160:864]

                else:
                    resampled_seg_frame = raw_frame

                #Apply mask to seg
                circular_mask = create_circular_mask(resampled_seg_frame.shape[0], resampled_seg_frame.shape[1], radius=346)
                masked_resampled_frame = np.invert(circular_mask) * resampled_seg_frame

                #Get only lipid
                lipid_array = np.zeros_like(masked_resampled_frame)
                lipid_array[masked_resampled_frame == 4] = 1

                #Sanity checks
                if np.isnan(lipid_array).any():
                    raise ValueError('NaN detected')
                
                unique_raw = np.unique(raw_frame)
                unique_new = np.unique(lipid_array)

                check_uniques(unique_raw, unique_new)
                    
                #Need to add extra dimension
                final_array = np.zeros((1, 704, 704))
                final_array[0,:,:] = lipid_array

                #Correct spacing and direction and save as nifti
                final_frame = sitk.GetImageFromArray(final_array.astype(np.uint32))
                final_frame.SetSpacing((1.0, 1.0, 999.0))
                final_frame.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                sitk.WriteImage(final_frame, final_path)


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)