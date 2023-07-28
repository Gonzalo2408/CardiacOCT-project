import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
sys.path.insert(1, '/mnt/netcache/diag/grodriguez/CardiacOCT/code/utils')
from conversion_utils import create_circular_mask, rgb_to_grayscale, sample_around

def main(argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/DICOM')
    parser.add_argument('--task', type=str, default='Task603_CardiacOCT')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--grayscale', type=bool, default=False)
    args, _ = parser.parse_known_args(argv)

    parent_path = args.data
    
    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')

    files = os.listdir(parent_path)

    #Frames we want to sample around annotation 
    # print('We are sampling {} frames before and after each annotation'.format(args.k))
    # if args.grayscale == True:
    #     print('Sampling grayscale')

    # else:
    #     print('Sampling RGB')

    for file in files:

        #Get image metadata from Excel file
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        #Output folder
        if belonging_set == 'Testing':

            output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTs'.format(args.task)

        else:
            
            output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTr'.format(args.task)

        #More metadata
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        pullback_name = file.split('.')[0]
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

        print('Reading image ', file)

        #Load the files to create a list of slices
        print('Loading DICOM...')
        series = sitk.ReadImage(parent_path +'/'+file)
        series_pixel_data = sitk.GetArrayFromImage(series)

        #Get frames with annotations in the pullback
        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

        # #Save grayscale images
        # if args.grayscale == True:

        #     #Convert to grayscale
        #     gray_img = rgb_to_grayscale(series_pixel_data)

        #     for frame in range(len(series_pixel_data)):
                
        #         if frame in frames_list:
                    
        #             count = 0
        #             frames_around = sample_around(gray_img, frame, args.k)

        #             for new_frame in range(frames_around.shape[2]):

        #                 final_path = output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id, "%04d" % (count))

        #                 if os.path.exists(final_path):
        #                     count += 1
        #                     print('File already exists')
        #                     continue
                                
        #                 else:

        #                     #Apply circular mask
        #                     circular_mask = create_circular_mask(frames_around[:,:,new_frame].shape[0], frames_around[:,:,new_frame].shape[1], radius=346)
        #                     mask_channel = np.invert(circular_mask) * frames_around[:,:,new_frame]

        #                     #Check if there are Nan values
        #                     if np.isnan(mask_channel).any():
        #                         raise ValueError('NaN detected')

        #                     final_image = sitk.GetImageFromArray(mask_channel)
        #                     final_image.SetSpacing((1.0, 1.0, 999.0))
        #                     final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        #                     sitk.WriteImage(final_image, final_path)
        #                     count += 1

        #RGB case
    
        for n_channel in range(3):

            print('Channel ', n_channel+1)

            for frame in range(len(series_pixel_data)):

                if frame in frames_list:

                    count = 0
                    frames_around = sample_around(series_pixel_data[:,:,:,n_channel], frame, args.k)

                    for new_frame in range(frames_around.shape[2]):
                        
                        final_path = output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id, "%04d" % (count+n_channel))

                        if os.path.exists(final_path):
                            print('File already exists')
                            count += 3
                            continue

                        else:

                            #Apply circular mask
                            circular_mask = create_circular_mask(frames_around[:,:,new_frame].shape[0], frames_around[:,:,new_frame].shape[1], radius=346)
                            mask_channel = np.invert(circular_mask) * frames_around[:,:,new_frame]

                            #Check if there are Nan values
                            if np.isnan(mask_channel).any():
                                raise ValueError('NaN detected')

                            final_image_after = sitk.GetImageFromArray(mask_channel)
                            final_image_after.SetSpacing((1.0, 1.0, 999.0))
                            final_image_after.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(final_image_after, final_path)
                            count += 3

        print('Done. Saved {} frames from pullback {} \n'.format(len(frames_list), pullback_name))
        print('###########################################\n')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)