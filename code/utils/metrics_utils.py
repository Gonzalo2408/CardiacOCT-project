import numpy as np
import math


def mean_metrics(list_dicts):
    """Gets the mean DICE of the total pullback for each label

    Args:
        list_dicts (list of dicts): list in which each element is a nested dictionary. Each label is a key of the first dictionary
        and the value is another dictionary containing all metrics (key: name of metric, value: metric value)

    Returns:
        dict: Dictionary that has a label as key and a list of the DICE for every frame in the pullback as value
    """    

    result = {}
    for d in list_dicts:
        for label, metrics in d.items():

            #Replace NaN to string so it can be loaded in Excel
            if math.isnan(metrics['Dice']):
                result[label] = 'NaN'

            else:
                result[label] = metrics['Dice']
            
    return result


def calculate_confusion_matrix(Y_true, Y_pred, labels):
    """Obtain confusion matrix for full pullback

    Args:
        Y_true (np.array): Array of the original image
        Y_pred (np.array): Array of the predicted image
        labels (list): range containing the number of classes

    Returns:
        np.array: array of shape (num_classes, num_classes)
    """    

    cm = np.zeros((len(labels), len(labels)), dtype=np.int)

    for i, x in enumerate(labels):
        for j, y in enumerate(labels):

            cm[i, j] = np.sum((Y_true == x) & (Y_pred == y))

    return cm

def dice_from_cm(cm):
    """Get DICE score from the previously obtained confusion matrix

    Args:
        cm (np.array): confusion matrix with shape (num_classes, num_classes)

    Returns:
        np.array: array of length num classes containing all of the DICEs computed per pullback
    """    

    assert (cm.ndim == 2)
    assert (cm.shape[0] == cm.shape[1])

    dices = np.zeros((cm.shape[0]))

    for i in range(cm.shape[0]):
        dices[i] = 2 * cm[i, i] / float(np.sum(cm[i, :]) + np.sum(cm[:, i]))

    return dices