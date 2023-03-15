import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd

#parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/scans DICOM'

parent_path = r'Z:\grodriguez\CardiacOCT\data-original\scans DICOM'
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

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

def resize_image(raw_frame):

    if raw_frame.shape == (704, 704):

        resampled_seg_frame = raw_frame

    else:

        frame_image = sitk.GetImageFromArray(raw_frame)

        new_shape = (704, 704)
        new_spacing = (frame_image.GetSpacing()[0]*sitk.GetArrayFromImage(frame_image).shape[1]/704,
                            frame_image.GetSpacing()[1]*sitk.GetArrayFromImage(frame_image).shape[1]/704)

        resampler = sitk.ResampleImageFilter()

        resampler.SetSize(new_shape)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputSpacing(new_spacing)

        resampled_seg = resampler.Execute(frame_image)
        resampled_seg_frame = sitk.GetArrayFromImage(resampled_seg)

    return resampled_seg_frame


def main(argv):
    """Callable entry point.
    """

    files = os.listdir(parent_path)

    i = 0

    for filename in files:

        patient_name = "-".join(filename.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':
            #new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task503_CardiacOCT/imagesTs'
            new_path_imgs = r'Z:\grodriguez\CardiacOCT\data-3d\nnUNet_raw_data\Task504_CardiacOCT\imagesTs'

        else:
            #new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task503_CardiacOCT/imagesTr'
            new_path_imgs = r'Z:\grodriguez\CardiacOCT\data-3d\nnUNet_raw_data\Task504_CardiacOCT\imagesTr'

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

                    channel_pixel_data = np.zeros((series_pixel_data.shape[0], 704, 704))
                    print('Processing channel ', i+1)     

                    for frame in range(len(series_pixel_data)):     

                        resized_image_pixel_data = resize_image(series_pixel_data[frame,:,:,i])
                        
                        #Circular mask as preprocessing (remove Abbott watermark)
                        circular_mask = create_circular_mask(resized_image_pixel_data.shape[0], resized_image_pixel_data.shape[1], radius=346)

                        channel_pixel_data[frame,:,:] = np.invert(circular_mask) * resized_image_pixel_data

                        #Check if there are Nan values
                        if np.isnan(channel_pixel_data[frame,:,:]).any():
                            raise ValueError('NaN detected')

                    print('Writing NIFTI file...')
                    channel_pixel_data_T = np.transpose(channel_pixel_data, (1, 2, 0))
                    final_image = sitk.GetImageFromArray(channel_pixel_data_T.astype(np.uint8))
                    final_image.SetSpacing((1.0, 1.0, 1.0))
                    final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                    sitk.WriteImage(final_image, filename_path)
                    
            print('Done\n')
            print('###########################################\n')
            i += 1

            if i == 2:
                break

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)