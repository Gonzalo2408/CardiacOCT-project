import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse

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

def sample_around(image, frame, n_channel, k):

    neighbors = np.arange(frame-k, frame+k+1)

    frames_around = np.zeros((image.shape[1], image.shape[2], len(neighbors)))

    for i in range(len(neighbors)):

        if neighbors[i] < 0 or neighbors[i] >= image.shape[0]:

            frames_around[:,:,i] = np.zeros((image.shape[1], image.shape[2]))

        else:
            frames_around[:,:,i] = image[neighbors[i],:,:, n_channel]

    return frames_around

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/netcache/diag/grodriguez/CardiacOCT/data-original/scans-DICOM')
    parser.add_argument('--task', type=str, default='Task507_CardiacOCT')
    parser.add_argument('--k', type=int, default=2)
    args, _ = parser.parse_known_args(argv)

    parent_path = args.data
    
    
    annots = pd.read_excel('/mnt/netcache/diag/grodriguez/CardiacOCT/excel-files/train_test_split_final.xlsx')

    files = os.listdir(parent_path)

    #Frames we want to sample around annotation 
    print('We are sampling {} frames before and {} frames after each annotation'.format(args.k, args.k))

    for file in files:

        #Check image count so image and corresponding seg id are the same(cluster reads images randomly, so cannot use counter)
        patient_name = "-".join(file.split('.')[0].split('-')[:3])
        belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

        if belonging_set == 'Testing':

            output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTs'.format(args.task)
            #output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTs'.format(args.task)

        else:
            
            output_file_path = 'Z:/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTr'.format(args.task)
            #output_file_path = '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/nnUNet_raw_data/{}/imagesTr'.format(args.task)

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

                    count = 0

                    frames_around = sample_around(series_pixel_data, frame, n_channel, args.k)

                    for new_frame in range(frames_around.shape[2]):

                        if os.path.exists(output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id, "%04d" % (count+n_channel))):
                            print('File already exists')
                            count += 3
                            continue

                        else:
                        
                            #Apply circular mask
                            circular_mask = create_circular_mask(frames_around[:,:,new_frame].shape[0], frames_around[:,:,new_frame].shape[1], radius=346)
                            mask_channel = np.invert(circular_mask) * frames_around[:,:,new_frame]

                            #Check if there are Nan values
                            if np.isnan(mask_channel).any():
                                raise ValueError('NaN detected')

                            final_image_after = sitk.GetImageFromArray(mask_channel)
                            final_image_after.SetSpacing((1.0, 1.0, 999.0))
                            final_image_after.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(final_image_after, output_file_path + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_{}.nii.gz'.format(n_pullback, frame, "%03d" % id, "%04d" % (count+n_channel)))
                            count += 3

        print('Done. Saved {} frames from pullback {} \n'.format(len(frames_list), pullback_name))
        print('###########################################\n')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)