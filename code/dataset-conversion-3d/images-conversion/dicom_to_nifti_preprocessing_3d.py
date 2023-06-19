import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
from utils.conversion_utils import create_circular_mask

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans-DICOM')
    parser.add_argument('--task', type=str, default='Task512_CardiacOCT')
    args, _ = parser.parse_known_args(argv)

    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/info-files/train_test_split_final.xlsx')


    for filename in os.listdir(args.data):

        #Get file metadata from Excel
        patient_name = "-".join(filename.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        #Output folder
        if belonging_set == 'Testing':
            new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/{}/imagesTs'.format(args.task)

        else:
            new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/{}/imagesTr'.format(args.task)

        #More data from file
        pullback_name = filename.split('.')[0]
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])


        #Load the files to create a list of slices
        print('Loading DICOM...')
        series = sitk.ReadImage(args.data +'/'+filename)
        series_pixel_data = sitk.GetArrayFromImage(series)

        for n_channel in range(3):

            final_path = os.path.join(new_path_imgs, patient_name + '_{}_{}_000{}.nii.gz'.format(n_pullback, "%03d" % id, n_channel))

            #Check if one of the channels already exist
            if os.path.exists(final_path):
                print('Channel {} already exists'.format(i+1))
                continue

            else:

                print('Channel ', n_channel+1)
                channel_pixel_data = np.zeros((series_pixel_data.shape[0], 704, 704))

                for frame in range(len(series_pixel_data)):

                    #Apply circular mask (remove Abbott watermark)
                    circular_mask = create_circular_mask(series_pixel_data[frame,:,:,n_channel].shape[0], series_pixel_data[frame,:,:,n_channel].shape[1], radius=346)
                    channel_pixel_data[frame,:,:] = np.invert(circular_mask) * series_pixel_data[frame,:,:,n_channel]

                    #Check if there are Nan values
                    if np.isnan(channel_pixel_data[frame,:,:]).any():
                        raise ValueError('NaN detected')

                print('Writing NIFTI file...')
                channel_pixel_data_T = np.transpose(channel_pixel_data, (1, 2, 0))
                final_image = sitk.GetImageFromArray(channel_pixel_data_T.astype(np.uint8))
                final_image.SetSpacing((1.0, 1.0, 1.0))
                final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                sitk.WriteImage(final_image, final_path)

            print('Done\n')
            print('###########################################\n')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)