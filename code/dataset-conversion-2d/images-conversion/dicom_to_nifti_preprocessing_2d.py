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
    ##### Paths for first dataset (cluster and my PC) #####
    #parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans DICOM'
    #annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split.xlsx')

    #parent_path = r'Z:\grodriguez\CardiacOCT\data-original\scans DICOM'
    #annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split.xlsx')


    ##### Paths for second dataset (cluster and my PC) #####
    #parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/extra scans DICOM'
    #annots = pd.read_excel(r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

    #parent_path = r'Z:\grodriguez\CardiacOCT\data-original\extra scans DICOM'
    #annots = pd.read_excel(r'Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')


    ##### Paths for third dataset (cluster and my PC)
    #parent_path = r'Z:\grodriguez\CardiacOCT\data-original\scans DICOM'
    #annots = pd.read_excel(r'Z:/grodriguez/CardiacOCT/data-original/train_test_split_final.xlsx')

    parent_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans DICOM'
    annots = pd.read_excel(r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/train_test_split_final.xlsx')

    files = os.listdir(parent_path)

    for file in files:

        #Check image count so image and corresponding seg id are the same(cluster reads images randomly, so cannot use counter)
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':

            #output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task503_CardiacOCT/imagesTs'
            output_file_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task503_CardiacOCT/imagesTs'

        else:
            #output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task503_CardiacOCT/imagesTr'
            output_file_path = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task503_CardiacOCT/imagesTr'

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

                    raw_frame = series_pixel_data[frame,:,:,n_channel]

                    #Resize image to (704, 704)
                    resampled_dcm_frame = resize_image(raw_frame)

                    #Apply circular mask
                    circular_mask = create_circular_mask(resampled_dcm_frame.shape[0], resampled_dcm_frame.shape[1], radius=346)
                    mask_channel = np.invert(circular_mask) * resampled_dcm_frame

                    #Check if there are Nan values
                    if np.isnan(mask_channel).any():
                        raise ValueError('NaN detected')


                    final_image = sitk.GetImageFromArray(mask_channel)
                    final_image.SetSpacing((999.0, 1.0, 1.0))
                    final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

                    sitk.WriteImage(final_image, output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_000{}.nii.gz'.format(n_pullback, frame, "%03d" % id, n_channel))

        print('Done. Saved {} frames from pullback {} \n'.format(len(frames_list), pullback_name))
        print('###########################################\n')
if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)