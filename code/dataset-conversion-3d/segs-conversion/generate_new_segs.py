import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from skimage.transform import resize

#path_segs = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/segmentations ORIGINALS'
#new_path_segs = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/nnUNet_raw_data/Task502_CardiacOCT/labelsTr'
#annots = pd.read_excel(r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/oct_annotations_filtered.xlsx')

path_segs = 'Z:/grodriguez/CardiacOCT/data/segmentations ORIGINALS'
new_path_segs = 'Z:/grodriguez/CardiacOCT/data/nnUNet_raw_data/Task502_CardiacOCT/labelsTr'
annots = pd.read_excel('Z:\grodriguez\CardiacOCT\data\oct_annotations_filtered.xlsx')

for filename in os.listdir(path_segs):

    new_filename = filename.replace('-', '').split('.')[0]

    id = annots.loc[annots['Pullback'] == filename.split('.')[0]]['ID'].values[0]

    print(new_filename+'_'+"%03d" % id+'.nii.gz')

    #Check that seg already exists
    if os.path.exists(new_path_segs + '/' + new_filename+'_'+ "%03d" % id + '.nii.gz'):
        print('Already exists. Skip')
        continue

    else:
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
        name = filename.split('.')[0]
        frames_with_annot = annots.loc[annots['Pullback'] == name]['Frames']

        frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
        print(frames_list)

        #If frame has manual annotations continue, else convert that frame to -1 (unlabeled data)
        for frame in range(len(resampled_pixel_data)):
            if frame in frames_list:
                continue
            else:
                 resampled_pixel_data[frame, :, :] = 0

        #Check if there are nans or empty frames (all labels 0)
        for frame in range(len(resampled_pixel_data)):

            if np.isnan(resampled_pixel_data[frame, :, :]).any():
                raise ValueError('NaN detected')

            # if np.all((resampled_pixel_data[frame,:,:] == 0)):
            #     raise ValueError('Labels disappeared. There are empty frames')

        #Fix spacing and direction
        final_seg = sitk.GetImageFromArray(resampled_pixel_data)
        final_seg.SetSpacing((1.0, 1.0, 1.0))
        final_seg.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        sitk.WriteImage(final_seg, new_path_segs + '/' + new_filename+'_'+ "%03d" % id + '.nii.gz')

