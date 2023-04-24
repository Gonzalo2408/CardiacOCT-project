import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')
path_segs = 'Z:/grodriguez/CardiacOCT/data-original/segmentations-ORIGINALS'
#annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

def generate_new_volume(image, n_frame, n_frames_to_sample = 1):

    # TO DO##
    # Add the case in which two annotated frames are very close to each other.
    # In that case, I think we should consider the n annotated frames in the same subvolume

    rows, cols, n_slices = image.shape

    #Check if annot is in first slice (we take frames after only)
    if n_frame - n_frames_to_sample <= 0:
        frames = np.arange(n_frames_to_sample, n_frames_to_sample+3)
        
    #Check if annot is at the end of the 3D volume
    elif n_frame + n_frames_to_sample >= n_slices:
        frames = np.arange(-n_frames_to_sample-2, n_frames_to_sample)

    else:
        frames = np.arange(-n_frames_to_sample, n_frames_to_sample+1)

    frames_to_sample = frames + n_frame
    sub_volume = np.zeros((rows, cols, len(frames_to_sample)))

    for i in range(len(frames_to_sample)):

        sub_volume[:, :, i] = image[:, :, frames_to_sample[i]]

    return sub_volume, frames_to_sample

def generate_cluster_volume(image, n_frame, frames_list, list_skips, k=30):

    #K-neighbors to find near annotations
    rows, cols, n_slices = image.shape
    neighbor_annots = np.arange(n_frame-k, n_frame+k)
    list_neighbors = []
    
    for neighbor in neighbor_annots:
        
        if neighbor in frames_list:
            list_neighbors.append(neighbor)
            list_skips.append(neighbor)
            
        else:
            continue

    print('Frame {} has {} neighbor(s). The new volume will have the annotations in {}'.format(n_frame, len(list_neighbors)-1, list_neighbors))

    #Case in which the last annotation is also 2 frames away from the end of the full 3D scan
    if list_neighbors[-1] + 10 > n_slices:
        frames_to_sample = np.arange(list_neighbors[0]-10, list_neighbors[-1]+1)
    
    #Case in which an annotation is the first one
    elif list_neighbors[0] - 10 < 0:
        frames_to_sample = np.arange(list_neighbors[0], list_neighbors[-1]+11)

    #Case in which both cases occur
    elif list_neighbors[-1] + 10 > n_slices and list_neighbors[0] - 10 < 0:
        frames_to_sample = np.arange(list_neighbors[0], list_neighbors[-1])

    #We take all frames that contain the annotations plus 1 frames before and after the cluster of slices
    else:
        frames_to_sample = np.arange(list_neighbors[0]-10, list_neighbors[-1]+11)

    print('We are sampling these frames ', frames_to_sample)

    sub_volume = np.zeros((rows, cols, len(frames_to_sample)))

    for i in range(len(frames_to_sample)):
        
        sub_volume[:, :, i] = image[:, :, frames_to_sample[i]]

    print('A sub_volume has been generated with shape {}'.format(sub_volume.shape))

    return sub_volume, frames_to_sample, list_neighbors


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


def check_uniques(raw_unique, new_unique, frame):
    if len(raw_unique) != len(new_unique):
        print(raw_unique, new_unique)
        print('Warning! There are noisy pixel values in the frame {}. Check resampling technique or image generated'.format(frame))
        return False

    for i in range(len(raw_unique)):
        if raw_unique[i] != new_unique[i]:
            print(raw_unique, new_unique)
            print('Warning! There are noisy pixel values in the frame {}. Check resampling technique or image generated'.format(frame))
            return False

    return True

def main(argv):

    a = 0

    split_data_pd = pd.DataFrame(columns = ['Pullback', 'Shape volume', 'Nº split', 
                                        'Starting frame', 
                                        'Ending frame', 'Annotated frames'])

    for filename in os.listdir(path_segs):

        #Geting patient ID from pullback
        patient_name = "-".join(filename.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':
            new_path_segs = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task505_CardiacOCT/labelsTs'

        else:
            new_path_segs = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task505_CardiacOCT/labelsTr'

        pullback_name = filename.split('.')[0]
        print(pullback_name)
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

        frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

        print('There is a total of {} annotations. Sampling around annotations...'. format(len(frames_list)))


        #Check that file already exists
        stop = False
        for file in os.listdir(new_path_segs):
            if '{}_{}'.format(patient_name, n_pullback) in file:
                print('Patient already processed. Skip')
                stop = True
                break
            
            else: 
                continue
            
        if stop == True:
            continue

        else:
            orig_seg = sitk.ReadImage(path_segs + '/' + filename)
            orig_seg_pixel_array = sitk.GetArrayFromImage(orig_seg)
            spacing = annots.loc[annots['Pullback'] == pullback_name]['Spacing'].values[0]

            frame_data = np.zeros((orig_seg_pixel_array.shape[0], 704, 704))

            for frame in range(len(frame_data)):

                if frame in frames_list:

                    #Check if resize is neeeded (shape should be (704, 704))
                    if orig_seg_pixel_array[frame,:,:].shape == (1024, 1024) and spacing == 0.006842619:
                        resampled_pixel_data = resize_image(orig_seg_pixel_array[frame,:,:])

                    elif orig_seg_pixel_array[frame,:,:].shape == (1024, 1024) and spacing == 0.009775171:
                        resampled_pixel_data = orig_seg_pixel_array[frame,:,:][160:864, 160:864]

                    elif orig_seg_pixel_array[frame,:,:].shape == (704, 704) and (spacing == 0.014224751 or spacing == 0.014935988):
                        resampled_pixel_data = resize_image(orig_seg_pixel_array[frame,:,:], False)
                        resampled_pixel_data = resampled_pixel_data[160:864, 160:864]

                    else:
                        resampled_pixel_data = orig_seg_pixel_array[frame,:,:]

                    circular_mask = create_circular_mask(resampled_pixel_data.shape[0], resampled_pixel_data.shape[1], radius=346)
                    mask_frame = np.invert(circular_mask) * resampled_pixel_data

                    unique_raw = np.unique(orig_seg_pixel_array[frame,:,:])
                    unique_new = np.unique(mask_frame)
                    check_uniques(unique_raw, unique_new, frame)

                else:

                    mask_frame = -1*np.zeros((704, 704))

                if np.isnan(mask_frame).any():
                    raise ValueError('NaN detected')

                frame_data[frame,:,:] = mask_frame

            frame_data_T = np.transpose(frame_data, (1, 2, 0))

            n_split = 1

            #Find neighbor annotations to create clusters
            list_skips = []

            for frame in frames_list:

                #List that contains annots that already have been included (avoid redundant data)
                if frame in list_skips:
                    continue

                else:
                    
                    sub_volume, frames_to_sample = generate_new_volume(frame_data_T, frame)

                    # #Get only "big" volumes
                    # if sub_volume.shape[2] < 30:
                    #     continue
                    
                    # else:

                    #Fix spacing and direction
                    final_seg = sitk.GetImageFromArray(sub_volume.astype(np.int32))
                    final_seg.SetSpacing((1.0, 1.0, 1.0))
                    final_seg.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

                    sitk.WriteImage(final_seg, new_path_segs + '/' + patient_name + '_{}_split{}_{}.nii.gz'.format(n_pullback, n_split, "%03d" % id ))
                    
                    #Create new Excel file to see how is the sampling working
                    frame_info_to_pd = []
                    frame_info_to_pd.append(pullback_name)
                    frame_info_to_pd.append(sub_volume.shape)
                    frame_info_to_pd.append(n_split)
                    frame_info_to_pd.append(frames_to_sample[0])
                    frame_info_to_pd.append(frames_to_sample[-1])
                    frame_info_to_pd.append(len(frames_to_sample))
                    
                    split_data_pd = split_data_pd.append(pd.Series(frame_info_to_pd, index=split_data_pd.columns[:len(frame_info_to_pd)]), ignore_index=True)

                    n_split += 1

        a += 1
        if a > 2:
            break

    split_data_pd.to_excel(r'Z:\grodriguez\CardiacOCT\excel-files\pseudo_volumes_data.xlsx')


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)