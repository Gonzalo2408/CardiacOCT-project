import numpy as np
import os


def create_image_png(segmentation: np.array) -> np.array:
    """Returns the raw segmentation and the measruements as an overlay, using a fixed color mapping

        segmentaiton (np.array): Array cotaining the raw segmentation frame

    Returns:
        np.array: New array with the correcte color mapping
    """    

    #Specify color map
    color_map = {
    0: (0, 0, 0),        #background
    1: (255, 0, 0),      #lumen
    2: (0, 255, 0),      #guide
    3: (0, 0, 255),      #initma
    4: (255, 255, 0),    #lipid
    5: (255, 255, 255),  #calcium
    6: (255, 0, 255),    #media
    7: (146, 0, 0),      #catheter
    8: (255, 123, 0),    #sidebranch
    9: (230, 141, 230),  #rt
    10: (0, 255, 255),   #wt
    11: (65, 135, 100),  #scad
    12: (208, 190, 161), #rupture
    }

    #Convert the labels array into a color-coded image
    h, w = segmentation.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_img[segmentation == label] = color

    return color_img


def merge_frames_into_pullbacks(path_predicted: str) -> dict:
    """Creates a dictionary with the pullback as key and a list of the frames with annotations as value. The values in the list
       are just the filenames of the frames in the predicted data folder

    Args:
        path_predicted (str): Path of the predicted data

    Returns:
        dict: Dictionary
    """    

    #Get only nifti files in the pred folder
    pullbacks_origs = [i for i in os.listdir(path_predicted) if '.nii.gz' in i]
    pullbacks_origs_set = []
    pullbacks_dict = {}

    #Save into a list the patiend id + n_pullback substring
    for i in range(len(pullbacks_origs)):
        if pullbacks_origs[i].split('_frame')[0] not in pullbacks_origs_set:
            pullbacks_origs_set.append(pullbacks_origs[i].split('_frame')[0])

        else:
            continue

    #Create dict with patient_id as key and list of belonging frames as values
    for i in range(len(pullbacks_origs_set)):
        frames_from_pullback = [frame for frame in pullbacks_origs if pullbacks_origs_set[i] in frame]
        pullbacks_dict[pullbacks_origs_set[i]] = frames_from_pullback

    return pullbacks_dict


