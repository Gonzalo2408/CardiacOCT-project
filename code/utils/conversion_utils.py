import numpy as np
import SimpleITK as sitk
import cv2

def create_circular_mask(h, w, center=None, radius=None):
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



def resize_image(raw_frame, downsample = True):
    """Resize image to (704, 704) and also checks for wrong spacing

    Args:
        raw_frame (np.array): image to be resized
        downsample (bool, optional): this checks for the case in which the shape is wrong but the spacing is correct (check script to understand better). Defaults to True.

    Returns:
        np.array: _description_
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



def rgb_to_grayscale(img): 
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



def check_uniques(raw_unique, new_unique, frame):
    """We looked for weird labels that appeared in the segmentation during the conversion

    Args:
        raw_unique (_type_): unique labels of the original segmentation
        new_unique (_type_): unique labels of the processed segmentation

    Returns:
        _type_: _description_
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


def sample_around(image, frame, k):
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


def cartesian_to_polar(image):
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