import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from skimage.transform import resize

#path_segs = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/segmentations ORIGINALS'
#new_path_segs = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/nnUNet_raw_data/Task502_CardiacOCT/labelsTr'
#annots = pd.read_excel(r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/oct_annotations_filtered.xlsx')

path_segs = 'Z:/grodriguez/CardiacOCT/data-original/extra segmentations ORIGINALS'
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

list_ids = []

for filename in os.listdir(path_segs):

    #Geting patient ID from pullback
    patient_name = "-".join(filename.split('.')[0].split('-')[:3])
    belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

    if belonging_set == 'Testing':
        new_path_segs = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task502_CardiacOCT/labelsTs'

    else:
        new_path_segs = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/Task502_CardiacOCT/labelsTr'

    pullback_name = filename.split('.')[0]
    print(pullback_name)
    id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
    n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

    # if id not in list_ids:
    #     n_pullback = 1
    #     list_ids.append(id)

    # else:
    #     n_pullback = 2
    #     print('Found extra pullback for patient ', patient_name)

    orig_seg = sitk.ReadImage(path_segs + '/' + filename)
    orig_seg_pixel_array = sitk.GetArrayFromImage(orig_seg)

    #Check if resize or not resize image (shape should be (704, 704))
    if orig_seg_pixel_array[0,:,:].shape == (704, 704):
        resampled_seg = orig_seg

    else:

        new_shape = (704, 704, orig_seg_pixel_array.shape[0])
        new_spacing = (orig_seg.GetSpacing()[0]*sitk.GetArrayFromImage(orig_seg).shape[1]/704,
                            orig_seg.GetSpacing()[1]*sitk.GetArrayFromImage(orig_seg).shape[1]/704,
                            orig_seg.GetSpacing()[2])

        resampler = sitk.ResampleImageFilter()

        resampler.SetSize(new_shape)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(new_spacing)

        resampled_seg = resampler.Execute(orig_seg)

    resampled_pixel_data = sitk.GetArrayFromImage(resampled_seg).astype(np.int32)

    # Find the frames with annotations
    frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']

    frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
    print(frames_list)

    #We want frames with manual annotations, the throw the rest away
    for frame in range(len(resampled_pixel_data)):
        if frame in frames_list:

            #Check that a seg has already been generated
            if os.path.exists(new_path_segs + '/' + patient_name.replace('-', '') + '_{}_frame{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id)):
                print('File already exists. Skip')
                continue

            #Check for nans
            if np.isnan(resampled_pixel_data[frame, :, :]).any():
                raise ValueError('NaN detected')

            #Need to add extra dimension
            final_array = np.zeros((1, resampled_pixel_data.shape[1], resampled_pixel_data.shape[2]))
            final_array[0,:,:] = resampled_pixel_data[frame,:,:]

            #Fix spacing and direction
            final_seg = sitk.GetImageFromArray(final_array)
            final_seg.SetSpacing((1.0, 1.0, 1.0))
            final_seg.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            sitk.WriteImage(final_seg, new_path_segs + '/' + patient_name.replace('-', '') + '_{}_frame{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id))
        
        else:
            continue