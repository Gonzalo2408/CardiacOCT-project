import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse

annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')
#annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

def generate_new_volume(image, n_frame, n_frames_to_sample = 2):

    ## TO DO##
    # Add the case in which two annotated frames are very close to each other.
    # In that case, I think we should consider the n annotated frames in the same subvolume

    rows, cols, n_slices = image.shape

    #Check if annot is in first slice (we take frames after only)
    if n_frame - n_frames_to_sample < 0:
        frames = np.arange(n_frame, n_frames_to_sample+3)
        frames_to_sample = frames
        
    #Check if annot is at the end of the 3D volume
    elif n_frame + n_frames_to_sample > n_slices:
        frames = np.arange((-n_frames_to_sample-3), n_frame)
        frames_to_sample = frames + n_frame

    else:
        frames = np.arange(-n_frames_to_sample, n_frames_to_sample+1)
        frames_to_sample = frames + n_frame

    sub_volume = np.zeros((rows, cols, len(frames_to_sample)))

    for i in range(len(frames_to_sample)):

        sub_volume[:, :, i] = image[:, :, frames_to_sample[i]]

    return sub_volume


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r'Z:\grodriguez\CardiacOCT\data-original\extra-scans-DICOM-2')
    args, unknown = parser.parse_known_args(argv)

    files = os.listdir(args.data)

    for filename in files:

        patient_name = "-".join(filename.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':
            #new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task504_CardiacOCT/imagesTs'
            new_path_imgs = r'Z:\grodriguez\CardiacOCT\data-3d\nnUNet_raw_data\Task505_CardiacOCT\imagesTs'

        else:
            #new_path_imgs = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task504_CardiacOCT/imagesTr'
            new_path_imgs = r'Z:\grodriguez\CardiacOCT\data-3d\nnUNet_raw_data\Task505_CardiacOCT\imagesTr'


        pullback_name = filename.split('.')[0]
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

        #Load the files to create a list of slices
        
        series = sitk.ReadImage(args.data +'/'+filename)
        series_pixel_data = sitk.GetArrayFromImage(series)
        print('Loading DICOM {} ...'.format(filename))

        for i in range(3):

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

            n_split = 1

            for frame in range(len(channel_pixel_data_T)):

                if frame in frames_list:

                    if frame - 2 < 0:
                        continue

                    else:
                        sub_volume = generate_new_volume(channel_pixel_data_T, frame)

                    #Fix spacing and direction
                    final_image = sitk.GetImageFromArray(sub_volume.astype(np.uint8))
                    final_image.SetSpacing((1.0, 1.0, 1.0))
                    final_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                    filename_path = new_path_imgs + '/' + patient_name + '_{}_{}_split{}_000{}.nii.gz'.format(n_pullback, "%03d" % id, n_split, i)

                    sitk.WriteImage(final_image, filename_path)

                    n_split += 1

                else:
                    continue

        print('Done\n')
        print('###########################################\n')
        break

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)