import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
sys.path.insert(1, "/mnt/netcache/diag/grodriguez/CardiacOCT/code/utils")
from conversion_utils import create_circular_mask, rgb_to_grayscale


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/DICOM')
    parser.add_argument('--task', type=str)
    parser.add_argument('--grayscale', action='store_true')
    args, _ = parser.parse_known_args(argv)

    parent_path = args.data

    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')

    files = os.listdir(parent_path)

    for file in files:

        # Get file data from the metadata Excel file
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        # Folder to stored the converted data
        if belonging_set == 'Testing':
            output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTs'.format(args.task)

        else:
            output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTr'.format(args.task)

        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        pullback_name = file.split('.')[0]
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])
        print('Reading image ', file)

        print('Loading DICOM...')
        series = sitk.ReadImage(os.path.join(parent_path, file))
        series_pixel_data = sitk.GetArrayFromImage(series)

        # Get the frames that will be processed
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

        # Do grayscale conversion
        if args.grayscale:

            # Convert to grayscale
            gray_img = rgb_to_grayscale(series_pixel_data)

            for frame in range(len(gray_img)):

                if frame in frames_list:

                    final_path = os.path.join(output_file_path,
                                              '{}_{}_frame{}_{}_0000.nii.gz'.format(patient_name.replace("-", ""),
                                                                                    n_pullback, frame, "%03d" % id))

                    # Check for existing data
                    if os.path.exists(final_path):
                        print('File already exists')
                        continue

                    # Apply circular mask
                    circular_mask = create_circular_mask(gray_img[frame, :, :].shape[0], gray_img[frame, :, :].shape[1],
                                                         radius=346)
                    mask_channel = np.invert(circular_mask) * gray_img[frame, :, :]

                    # Check if there are Nan values
                    if np.isnan(mask_channel).any():
                        raise ValueError('NaN detected')

                    # Correct spacing and direction and save as nifti
                    final_image = sitk.GetImageFromArray(mask_channel)
                    final_image.SetSpacing((1.0, 1.0, 999.0))
                    final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

                    sitk.WriteImage(final_image, final_path)

        # To store RGB images
        else:

            for n_channel in range(3):

                print('Channel ', n_channel+1)

                for frame in range(len(series_pixel_data)):

                    if frame in frames_list:

                        final_path = os.path.join(output_file_path,
                                                  '{}_{}_frame{}_{}_000{}.nii.gz'.format(patient_name.replace("-", ""),
                                                                                         n_pullback, frame,
                                                                                         "%03d" % id, n_channel))

                        # Check for existing data
                        if os.path.exists(final_path):
                            print('File already exists')
                            continue

                        raw_frame = series_pixel_data[frame, :, :, n_channel]

                        # Apply circular mask
                        circular_mask = create_circular_mask(raw_frame.shape[0], raw_frame.shape[1], radius=346)
                        mask_channel = np.invert(circular_mask) * raw_frame

                        # Check if there are Nan values
                        if np.isnan(mask_channel).any():
                            raise ValueError('NaN detected')

                        # Correct spacing and direction and save as nifti
                        final_image = sitk.GetImageFromArray(mask_channel)
                        final_image.SetSpacing((1.0, 1.0, 999.0))
                        final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

                        sitk.WriteImage(final_image, final_path)

        print('Done. Saved {} frames from pullback {} \n'.format(len(frames_list), pullback_name))
        print('###########################################\n')


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)
