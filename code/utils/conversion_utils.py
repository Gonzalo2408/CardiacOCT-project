import numpy as np
import SimpleITK as sitk
from typing import Tuple
import cv2
from matplotlib.colors import ListedColormap
import base64

def create_circular_mask(h: int, w: int, center: tuple = None, radius: int = None) -> np.array:
    """Apply the circular mask to remove the Abbott watermark

    Args:
        h (int): x dim in image
        w (int): y dim in image
        center (tuple, optional): coords of the center of the circular mask in the image. Defaults to None.
        radius (int, optional): radius of the created circular mask. Defaults to None.

    Returns:
        np.array: Maskee image
    """    

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    mask = np.expand_dims(mask,0)

    return mask



def resize_image(raw_frame: np.array, downsample: bool = True) -> np.array:
    """Resize image to (704, 704) and also checks for wrong spacing

    Args:
        raw_frame (np.array): image to be resized
        downsample (bool, optional): this checks for the case in which the shape is wrong but the spacing is correct (check script to understand better). Defaults to True.

    Returns:
        np.array: Resampled image
    """    

    frame_image = sitk.GetImageFromArray(raw_frame)

    if downsample == True:
        new_shape = (704, 704)

    else:
        new_shape = (1024, 1024)


    new_spacing = (frame_image.GetSpacing()[0]*sitk.GetArrayFromImage(frame_image).shape[1]/new_shape[0],
                        frame_image.GetSpacing()[1]*sitk.GetArrayFromImage(frame_image).shape[1]/new_shape[0])

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(new_shape)
    #We are using nearest neighbour interpolation
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)

    resampled_seg = resampler.Execute(frame_image)
    resampled_seg_frame = sitk.GetArrayFromImage(resampled_seg)

    return resampled_seg_frame



def rgb_to_grayscale(img: np.array) -> np.array: 
    """Converts RGB OCT scan to grayscale

    Args:
        img (np.array): Raw RGB OCT image

    Returns:
        np.array: Grayscale image
    """    

    frames, rows, cols, _ = img.shape
    gray_img = np.zeros((frames, rows, cols))

    for frame in range(frames):

        #Apply grayscale formula
        grayscale = 0.2989 * img[frame,:,:,0] + 0.5870 * img[frame,:,:,1] + 0.1140 * img[frame,:,:, 2]
        gray_img[frame, :, :] = grayscale.astype(int)

    return gray_img 


def decode_rgb() -> np.array:
    """String to decode the RGB values into the corresponding grayscale, provided by Abbott

    Returns:
        np.array: array with the 255 grayscale values with the corresponding RGB
    """    
    
    golden = b'AAAABQEACQEADQEADwEAEQIAEgIAFAIAFgIAGAMAGgMAGwMAHQMAHgMAIAQAIQQAIwQAJgQAJwUAKQUAKgUALAUALQYALwYAMAYAMwcBN' \
            b'QgBNggBOAkBOQoBOwoBPAsBPgwBQA0BQg4BQw4BRA8BRhABRxEBSREBShIBTBMBThQBUBUBURYBUhcBVBgBVRkBVxkBWBoBWhsBXBwCXh' \
            b'0CXx0CYB4CYh8CYyACZCECZiMCZyQCaSUCayYCbCcCbSgCbykCcCoCcisCcywCdC0Cdi4CeC8CejACezECfDMCfTQCfzUCgTYCgjcDgzg' \
            b'DhDkDhToDhzsDiTwDij0Diz4EjEAEjUEEj0IEkEMEkUQEkkUElEYFlkgFl0kFmEkFmUoFm0wFnE0GnU4GnlAGn1EGoVIGo1MHpFQIpVUI' \
            b'plcIp1gJqFoJqFwJqV0Jql8Kq2AKq2IKrGMLrWQLrmULr2cMr2kMsGsMsmwNs20Ns28NtHANtXIOtnMOt3QPt3YPuHcPunoQu3sQu3sQv' \
            b'HwRvX4Rvn8Sv4ASwIESwYITwoQUw4UUw4YVxIgWxYkWx4oXyIsYyIwYyY0Yyo8Zy5AZy5EZzZIazpMbz5Yb0Jcc0Zgc0pkd05oe1Jsf1Z' \
            b'0g1Z4g1p8h16Ai2KEj2KMl2aQm2qYn26co3Kgp3qop36sq4Kwr4a0s4a4t4rAu47Iv5LMw5bQx5bUy5rYz57g06Lk16bo36rs567w6674' \
            b'87L8+7MFA7MJC7cNF7cRG7cVI7sdL7shM78lP78pR78tT8MxV8M5X8c9Y8dFa8dJc8tNf8tRh89Vj89Zl89dn9Nlp9Nls9Npv9Npx9dt0' \
            b'9d139d569d599t+A9uCC9uGE9+KI9+KL9+KN9+OQ+OST+eSW+eWZ+eab+uee+uih+uij+umn++uq++ys++yv/O2z/O20/O62/O+5/O+7/' \
            b'PC+/PDB/PHD/PLG/PLI/PPK/PPN/PTP/PTR/PXU/PbW/PbY/Pfb/Pje/Png/Prj/Prl/Prn/Pvq/Pzt'
    
    #colormap = ListedColormap(colors=[tuple(row) for row in np.frombuffer(base64.b64decode(golden), dtype=np.ubyte).reshape((256, 3)) / 256.0])
    array = [tuple(row) for row in np.frombuffer(base64.b64decode(golden), dtype=np.ubyte).reshape((256, 3)) / 256.0]
    
    return array


def check_uniques(raw_unique: np.array, new_unique: np.array, frame: int) -> bool:
    """We looked for weird labels that appeared in the segmentation during the conversion

    Args:
        raw_unique (np.array): unique labels of the original segmentation
        new_unique (np.array): unique labels of the processed segmentation
        frame (int): frame we are currently checking

    Returns:
        bool: True if everything is fine
    """    
 
    #Both should contain same amount of labels
    if len(raw_unique) != len(new_unique):
        print(raw_unique, new_unique)
        print('Warning! There are noisy pixel values in the frame {}. Check resampling technique or image generated'.format(frame))
        return False
    
    #Labels should be the same
    for i in range(len(raw_unique)):
        if raw_unique[i] != new_unique[i]:
            print(raw_unique, new_unique)
            print('Warning! There are noisy pixel values in the frame {}. Check resampling technique or image generated'.format(frame))
            return False
        
    return True


def sample_around(image: np.array, frame, int, k: int) -> np.array:
    """Get k frames around specific frame with annotation

    Args:
        image (np.array): Frame with annotation
        frame (int): Nº frame we are looking
        k (int): Number of frame to sample before and after

    Returns:
        np.array: Volume that contains the frame with annotation in the middle and the k frames before and after
    """    

    #Get neighbouring frakes
    neighbors = np.arange(frame-k, frame+k+1)

    frames_around = np.zeros((image.shape[1], image.shape[2], len(neighbors)))

    for i in range(len(neighbors)):

        #Case in which annotation is the first or last frame of the pullback (append black frames in that case)
        if neighbors[i] < 0 or neighbors[i] >= image.shape[0]:

            frames_around[:,:,i] = np.zeros((image.shape[1], image.shape[2]))

        else:
            frames_around[:,:,i] = image[neighbors[i],:,:]

    return frames_around


def cartesian_to_polar(image: np.array) -> np.array:
    """Converts the OCT image/segmentation to polar coordinates (perform resizing and circular mask before this)

    Args:
        image (np.array): Array containing the OCT image/segmentation

    Returns:
        np.array: Converted array
    """    

    value_img = np.sqrt(((image.shape[0]/2.0)**2.0)+((image.shape[1]/2.0)**2.0))
    polar_img = cv2.linearPolar(image,(image.shape[0]/2, image.shape[1]/2), value_img, cv2.WARP_FILL_OUTLIERS)
    polar_img = polar_img.astype(np.uint32)

    return polar_img