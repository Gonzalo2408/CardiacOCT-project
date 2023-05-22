import SimpleITK as sitk
import os
import numpy as np
import pandas as pd

path_segs = 'Z:/grodriguez/CardiacOCT/data-original/new-pullback-seg'
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    mask = np.expand_dims(mask, 0)

    return mask

def resize_image(raw_frame, downsample = True):

    frame_image = sitk.GetImageFromArray(raw_frame)

    if downsample == True:
        new_shape = (704, 704)

    else:
        new_shape = (1024, 1024)


    new_spacing = (frame_image.GetSpacing()[0]*sitk.GetArrayFromImage(frame_image).shape[1]/new_shape[0],
                        frame_image.GetSpacing()[1]*sitk.GetArrayFromImage(frame_image).shape[1]/new_shape[0])

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(new_shape)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)

    resampled_seg = resampler.Execute(frame_image)
    resampled_seg_frame = sitk.GetArrayFromImage(resampled_seg)

    return resampled_seg_frame

def check_uniques(raw_unique, new_unique):

    #The unique number of labels before and after preprocessing should be identical (this function checks that)
 
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

for filename in os.listdir(path_segs):

    #Geting patient ID from pullback
    patient_name = "-".join(filename.split('.')[0].split('-')[:3])
    belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

    if belonging_set == 'Testing':
        new_path_segs = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task508_CardiacOCT/labelsTs'

    else:
        new_path_segs = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task508_CardiacOCT/labelsTr'

    pullback_name = filename.split('.')[0]
    print('Checking ', pullback_name)

    #Get ID and nº of pullback
    id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
    n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

    #Read segmentation file
    orig_seg = sitk.ReadImage(path_segs + '/' + filename)
    orig_seg_pixel_array = sitk.GetArrayFromImage(orig_seg)

    # Find the frames with annotations
    frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
    frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

    for frame in range(len(orig_seg_pixel_array)):

        if frame in frames_list:

            #Check that a seg has already been generated
            if os.path.exists(new_path_segs + '/' + patient_name.replace('-', '') + '_{}_frame{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id)):
                print('File already exists. Skip')
                continue

            raw_frame = orig_seg_pixel_array[frame,:,:]
            spacing = annots.loc[annots['Pullback'] == pullback_name]['Spacing'].values[0]

            #Check if resize is neeeded (shape should be (704, 704))
            if raw_frame.shape == (1024, 1024) and spacing == 0.006842619:
                resampled_seg_frame = resize_image(raw_frame)

            elif raw_frame.shape == (1024, 1024) and spacing == 0.009775171:
                resampled_seg_frame = raw_frame[160:864, 160:864]

            elif raw_frame.shape == (704, 704) and (spacing == 0.014224751 or spacing == 0.014935988):
                resampled_seg_frame = resize_image(raw_frame, False)
                resampled_seg_frame = resampled_seg_frame[160:864, 160:864]

            else:
                resampled_seg_frame = raw_frame

            #Apply mask to both seg and image
            circular_mask = create_circular_mask(resampled_seg_frame.shape[0], resampled_seg_frame.shape[1], radius=346)
            masked_resampled_frame = np.invert(circular_mask) * resampled_seg_frame

            #Sanity checks
            if np.isnan(masked_resampled_frame).any():
                raise ValueError('NaN detected')
            
            unique_raw = np.unique(raw_frame)
            unique_new = np.unique(masked_resampled_frame)

            check_uniques(unique_raw, unique_new)
                
            #Need to add extra dimension
            final_array = np.zeros((1, 704, 704))
            final_array[0,:,:] = masked_resampled_frame

            final_frame = sitk.GetImageFromArray(final_array.astype(np.uint32))
            final_frame.SetSpacing((1.0, 1.0, 999.0))
            final_frame.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            sitk.WriteImage(final_frame, new_path_segs + '/' + patient_name.replace('-', '') + '_{}_frame{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id))