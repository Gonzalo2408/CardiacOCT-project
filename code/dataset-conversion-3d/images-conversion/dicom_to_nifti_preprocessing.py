import sys
import os
import numpy as np
from skimage.transform import resize
import SimpleITK as sitk

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

def obtain_image_count(segs_dir, file):

    for i, s in enumerate(segs_dir):
        if s.find(file) != -1:
            image_count = s.replace('_','.').split('.')[1]
            return image_count

def main(argv):
    """Callable entry point.
    """
    parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/scans DICOM'
    segs_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/nnUNet_raw_data/Task502_CardiacOCT/labelsTr'
    output_file_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/nnUNet_raw_data/Task502_CardiacOCT/imagesTr'
    #parent_path = r'Z:\grodriguez\CardiacOCT\data\scans DICOM'
    #segs_path = r'Z:\grodriguez\CardiacOCT\data\nnUNet_raw_data\Task502_CardiacOCT\labelsTr'
    #output_file_path = r'Z:\grodriguez\CardiacOCT\data\nnUNet_raw_data\Task502_CardiacOCT\imagesTr'

    files = os.listdir(parent_path)
    segs = os.listdir(segs_path)

    for file in files:

        #Check image count so image and corresponding seg id are the same(cluster reads images randomly, so cannot use counter)
        image_count = obtain_image_count(segs, file.replace("-", "").split('.')[0])

        nifti_file_r = output_file_path + '/' + file.replace("-", "").split('.')[0] + '_{}'.format(image_count) + '_0000.nii.gz'
        nifti_file_g = output_file_path + '/' + file.replace("-", "").split('.')[0] + '_{}'.format(image_count) + '_0001.nii.gz'
        nifti_file_b = output_file_path + '/' + file.replace("-", "").split('.')[0] + '_{}'.format(image_count) + '_0002.nii.gz'

        print('Reading image ', file)

        if os.path.exists(nifti_file_r) and os.path.exists(nifti_file_g) and os.path.exists(nifti_file_b):
            print('Files already created. Skip')
            continue

        else:

            # load the files to create a list of slices
            print('Loading DICOM...')
            series = sitk.ReadImage(parent_path +'/'+file)
            series_pixel_data = sitk.GetArrayFromImage(series)

            for i in range(3):

                filename = output_file_path + '/' + file.replace("-", "").split('.')[0] + '_{}'.format(image_count) + '_000{}.nii.gz'.format(i)

                #Check if one of the channels already exist
                if os.path.exists(filename):
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

                    sitk.WriteImage(final_image, filename)
                    print('Done')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)