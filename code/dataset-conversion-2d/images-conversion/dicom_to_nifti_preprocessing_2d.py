import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd

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
    """Callable entry point.
    """
    ##### Paths for first dataset (cluster and my PC) #####
    #parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans DICOM'
    #annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split.xlsx')

    #parent_path = r'Z:\grodriguez\CardiacOCT\data-original\scans DICOM'
    #annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split.xlsx')


    ##### Paths for second dataset (cluster and my PC) #####
    parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/extra scans DICOM'
    annots = pd.read_excel(r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

    #parent_path = r'Z:\grodriguez\CardiacOCT\data-original\extra scans DICOM'
    #annots = pd.read_excel(r'Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

    files = os.listdir(parent_path)

    for file in files:

        #Check image count so image and corresponding seg id are the same(cluster reads images randomly, so cannot use counter)
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':

            #output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task502_CardiacOCT/imagesTs'
            output_file_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task502_CardiacOCT/imagesTs'

        else:
            #output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task502_CardiacOCT/imagesTr'
            output_file_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task502_CardiacOCT/imagesTr'

        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        pullback_name = file.split('.')[0]
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])
        print('Reading image ', file)

        #Load the files to create a list of slices
        print('Loading DICOM...')
        series = sitk.ReadImage(parent_path +'/'+file)
        series_pixel_data = sitk.GetArrayFromImage(series)

        for i in range(3):

            print('Channel ', i+1)

            #Check if resized is needed (all images should be (704, 704))
            if series_pixel_data[0,:,:].shape == (704, 704):
                print('Shape is {}. No resized needed'.format(series_pixel_data.shape))
                resized_image = series

            else:
                print('Reshaping image...')
                new_shape = (704, 704, series_pixel_data.shape[0])
                new_spacing = (series.GetSpacing()[0]*sitk.GetArrayFromImage(series).shape[1]/704,
                                series.GetSpacing()[1]*sitk.GetArrayFromImage(series).shape[1]/704,
                                series.GetSpacing()[2])

                resampler = sitk.ResampleImageFilter()

                resampler.SetSize(new_shape)
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetOutputSpacing(new_spacing)

                resized_image = resampler.Execute(series)

            resized_image_pixel_data = sitk.GetArrayFromImage(resized_image)

            #Circular mask as preprocessing (remove Abbott watermark)
            print('Creating circular mask...')
            mask_channel = np.zeros((resized_image_pixel_data.shape[1], resized_image_pixel_data.shape[2]))
            circular_mask = create_circular_mask(mask_channel.shape[0], mask_channel.shape[1], radius=340)

            frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']

            frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
            print(frames_list)

            for frame in range(len(mask_channel)):

                if frame in frames_list:

                    if os.path.exists(output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_000{}.nii.gz'.format(n_pullback, frame, "%03d" % id, i)):
                        print('File already exists')
                        continue

                    mask_channel = np.invert(circular_mask) * resized_image_pixel_data[frame,:,:,i]

                    #Check if there are Nan values
                    if np.isnan(mask_channel).any():
                        raise ValueError('NaN detected')

                    print('Writing NIFTI file...')
                    final_image = sitk.GetImageFromArray(mask_channel)
                    final_image.SetSpacing((1.0, 1.0, 1.0))
                    final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

                    sitk.WriteImage(final_image, output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_000{}.nii.gz'.format(n_pullback, frame, "%03d" % id, i))

        print('Done\n')
        print('###########################################\n')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)