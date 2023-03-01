import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from skimage.transform import resize
import sys

#path_segs = r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/segmentations ORIGINALS'
#annots = pd.read_excel(r'/mnt/netcache/diag/grodriguez/CardiacOCT/data/oct_annotations_filtered.xlsx')

path_segs = 'Z:/grodriguez/CardiacOCT/data-original/segmentations ORIGINALS'
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

    for filename in os.listdir(path_segs):

        #Geting patient ID from pullback
        patient_name = "-".join(filename.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':
            new_path_segs = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task503_CardiacOCT/labelsTs'

        else:
            new_path_segs = 'Z:/grodriguez/CardiacOCT/data-3d/nnUNet_raw_data/Task503_CardiacOCT/labelsTr'

        pullback_name = filename.split('.')[0]
        print(pullback_name)
        id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])
        n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])

        #Check that seg already exists
        if os.path.exists(new_path_segs + '/' + patient_name + '_{}_{}.nii.gz'.format(n_pullback, "%03d" % id)):
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

            resampled_pixel_data = sitk.GetArrayFromImage(resampled_seg).astype(np.uint8)

            # Find the frames with annotations
            frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']

            frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]
            print(frames_list)

            #If frame has manual annotations continue, else convert that frame to -1 (unlabeled data)
            # for frame in range(len(resampled_pixel_data)):
            #     if frame in frames_list:
            #         continue
            #     else:
            #         resampled_pixel_data[frame, :, :] = -1

            mask = np.zeros((resampled_pixel_data.shape[0], resampled_pixel_data.shape[1], resampled_pixel_data.shape[2]))
            circular_mask = create_circular_mask(mask.shape[1], mask.shape[2], radius=340)

            #Check if there are nans or empty frames (all labels 0)
            for frame in range(len(mask)):

                mask[frame,:,:] = np.invert(circular_mask) * resampled_pixel_data[frame,:,:]

                if np.isnan(mask[frame, :, :]).any():
                    raise ValueError('NaN detected')

                # if np.all((resampled_pixel_data[frame,:,:] == 0)):
                #     raise ValueError('Labels disappeared. There are empty frames')

            #Fix spacing and direction
            final_seg = sitk.GetImageFromArray(mask)
            final_seg.SetSpacing((1.0, 1.0, 1.0))
            final_seg.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

            sitk.WriteImage(final_seg, new_path_segs + '/' + patient_name + '_{}_{}.nii.gz'.format(n_pullback, "%03d" % id))


if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)