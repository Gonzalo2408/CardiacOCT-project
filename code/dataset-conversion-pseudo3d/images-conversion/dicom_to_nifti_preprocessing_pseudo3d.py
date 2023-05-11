import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    mask = np.expand_dims(mask,0)
    return mask


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans-DICOM')
    args, _ = parser.parse_known_args(argv)

    parent_path = args.data
    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

    files = os.listdir(parent_path)

    for file in files:

        #Check image count so image and corresponding seg id are the same(cluster reads images randomly, so cannot use counter)
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':

            #output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task506_CardiacOCT/imagesTs'
            output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task506_CardiacOCT/imagesTs'

        else:
            #output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task506_CardiacOCT/imagesTr'
            output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task506_CardiacOCT/imagesTr'

        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        pullback_name = file.split('.')[0]
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])
        print('Reading image ', file)

        #Load the files to create a list of slices
        print('Loading DICOM...')
        series = sitk.ReadImage(parent_path +'/'+file)
        series_pixel_data = sitk.GetArrayFromImage(series)

        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

        for n_channel in range(3):

            print('Channel ', n_channel+1)

            for frame in range(len(series_pixel_data)):

                if frame in frames_list:

                    if os.path.exists(output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_000{}.nii.gz'.format(n_pullback, frame, "%03d" % id, n_channel)):
                        print('File already exists')
                        continue

                    #Get frames before and after as modality
                    raw_frame = series_pixel_data[frame,:,:,n_channel]
                    
                    if frame == 0:
                        raw_frame_before = np.zeros((series_pixel_data[frame,:,:,n_channel].shape))
                        raw_frame_after = series_pixel_data[frame+1,:,:,n_channel]

                    elif frame + 1 == len(series_pixel_data):
                        raw_frame_before = series_pixel_data[frame-1,:,:,n_channel]
                        raw_frame_after = np.zeros((series_pixel_data[frame,:,:,n_channel].shape))

                    else:
                        raw_frame_before = series_pixel_data[frame-1,:,:,n_channel]
                        raw_frame_after = series_pixel_data[frame+1,:,:,n_channel]

                    #Apply circular mask
                    circular_mask = create_circular_mask(raw_frame.shape[0], raw_frame.shape[1], radius=346)
                    mask_channel = np.invert(circular_mask) * raw_frame
                    mask_channel_before = np.invert(circular_mask) * raw_frame_before
                    mask_channel_after = np.invert(circular_mask) * raw_frame_after

                    #Check if there are Nan values
                    if np.isnan(mask_channel).any() or np.isnan(mask_channel_before).any() or np.isnan(mask_channel_after).any():
                        raise ValueError('NaN detected')


                    final_image = sitk.GetImageFromArray(mask_channel)
                    final_image.SetSpacing((1.0, 1.0, 999.0))
                    final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                    sitk.WriteImage(final_image, output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_000{}.nii.gz'.format(n_pullback, frame, "%03d" % id, n_channel))

                    final_image_before = sitk.GetImageFromArray(mask_channel_before)
                    final_image_before.SetSpacing((1.0, 1.0, 999.0))
                    final_image_before.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                    sitk.WriteImage(final_image_before, output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_000{}.nii.gz'.format(n_pullback, frame, "%03d" % id, 3+n_channel))

                    final_image_after = sitk.GetImageFromArray(mask_channel_after)
                    final_image_after.SetSpacing((1.0, 1.0, 999.0))
                    final_image_after.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                    sitk.WriteImage(final_image_after, output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_000{}.nii.gz'.format(n_pullback, frame, "%03d" % id, 6+n_channel))

        print('Done. Saved {} frames from pullback {} \n'.format(len(frames_list), pullback_name))
        print('###########################################\n')

        
if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)