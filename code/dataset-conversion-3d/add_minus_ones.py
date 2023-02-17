import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

files = os.listdir()

for file in files:

    if file.split('.')[0] == 'pkl':
        continue

    else:
        preprocessed_img = np.load(file)['data']

        seg_map = preprocessed_img[0,:,:,:]

        new_img = np.zeros((preprocessed_img.shape))

        for frame in range(len(seg_map)):

            if np.all((seg_map[0,:,:,frame]) == 0):

                seg_map[0,:,:,frame] = 1

        new_img
