import sys
import os
import numpy as np
from skimage.transform import resize
import SimpleITK as sitk
import pandas as pd

#parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/scans DICOM'

parent_path = r'Z:\grodriguez\CardiacOCT\data-original\scans DICOM'
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

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

    files = os.listdir(parent_path)

    for filename in files:

        patient_name = "-".join(filename.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':
            #new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task503_CardiacOCT/imagesTs'
            new_path_imgs = r'Z:\grodriguez\CardiacOCT\data-3d\nnUNet_raw_data\Task503_CardiacOCT\imagesTs'

        else:
            #new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task503_CardiacOCT/imagesTr'
            new_path_imgs = r'Z:\grodriguez\CardiacOCT\data-3d\nnUNet_raw_data\Task503_CardiacOCT\imagesTr'

        pullback_name = filename.split('.')[0]
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

        nifti_file_r = new_path_imgs + '/' + patient_name + '_{}_{}_0000.nii.gz'.format(n_pullback, "%03d" % id)
        nifti_file_g = new_path_imgs + '/' + patient_name + '_{}_{}_0001.nii.gz'.format(n_pullback, "%03d" % id)
        nifti_file_b = new_path_imgs + '/' + patient_name + '_{}_{}_0002.nii.gz'.format(n_pullback, "%03d" % id)

        print('Reading image ', pullback_name)

        if os.path.exists(nifti_file_r) and os.path.exists(nifti_file_g) and os.path.exists(nifti_file_b):
            print('Files already created. Skip')
            continue

        else:

            # load the files to create a list of slices
            print('Loading DICOM...')
            series = sitk.ReadImage(parent_path +'/'+filename)
            series_pixel_data = sitk.GetArrayFromImage(series)

            for i in range(3):

                filename_path = new_path_imgs + '/' + patient_name + '_{}_{}_000{}.nii.gz'.format(n_pullback, "%03d" % id, i)

                #Check if one of the channels already exist
                if os.path.exists(filename_path):
                    print('Channel {} already exists'.format(i+1))
                    continue

                else:

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
                    mask_channel = np.zeros((resized_image_pixel_data.shape[0], resized_image_pixel_data.shape[1], resized_image_pixel_data.shape[2]))
                    circular_mask = create_circular_mask(mask_channel.shape[1], mask_channel.shape[2], radius=340)

                    for frame in range(len(mask_channel)):

                        mask_channel[frame,:,:] = np.invert(circular_mask) * resized_image_pixel_data[frame,:,:,i]

                        #Check if there are Nan values
                        if np.isnan(mask_channel[frame, :, :]).any():
                            raise ValueError('NaN detected')

                    print('Writing NIFTI file...')
                    final_image = sitk.GetImageFromArray(mask_channel)
                    final_image.SetSpacing((1.0, 1.0, 1.0))
                    final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

                    sitk.WriteImage(final_image, filename_path)
                    print('Done')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)