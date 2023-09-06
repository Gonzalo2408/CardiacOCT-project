import numpy as np
import math
from typing import List, Dict, Tuple


def mean_metrics(list_dicts: List[Dict[str, int]]) -> dict:
    """Gets the mean DICE of the total pullback for each label

    Args:
        list_dicts (list of dicts): list in which each element is a nested dictionary.
                                    Each label is a key of the first dictionary
        and the value is another dictionary containing all metrics (key: name of metric, value: metric value)

    Returns:
        dict: Dictionary that has a label as key and a list of the DICE for every frame in the pullback as value
    """

    labels_name = ['lumen', 'guidewire', 'wall', 'lipid', 'calcium', 'media', 'catheter', 'sidebranch',
                   'rthrombus', 'wthrombus', 'dissection', 'rupture']

    result = {}
    for d in list_dicts:
        for label, metrics in d.items():

            # Replace NaN to string so it can be loaded in Excel
            if math.isnan(metrics['Dice']):
                result[int(label)] = 'NaN'

            else:
                result[int(label)] = metrics['Dice']

    # Sort items
    result = dict(sorted(result.items()))

    # Change name of the labels (so instead of getting the label nº, you get the name)
    final_dict = {}

    for i, key in enumerate(result):
        final_dict[labels_name[i]] = result[key]

    return final_dict


def calculate_confusion_matrix(Y_true: np.array, Y_pred: np.array, labels: list) -> np.array:
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


def metrics_from_cm(cm: np.array) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Get metrics score the previously obtained confusion matrix

    Args:
        cm (np.array): confusion matrix with shape (num_classes, num_classes)

    Returns:
        tuple: tuple with arrays of length num classes containing all of the DICEs, precision,
        recall and specificity computed per pullback
    """

    assert (cm.ndim == 2)
    assert (cm.shape[0] == cm.shape[1])

    dices = np.zeros((cm.shape[0]))
    ppv = np.zeros((cm.shape[0]))
    npv = np.zeros((cm.shape[0]))
    sens = np.zeros((cm.shape[0]))
    spec = np.zeros((cm.shape[0]))
    kappa = np.zeros((cm.shape[0]))

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        dices[i] = 2 * tp / float(2 * tp + fp + fn)
        ppv[i] = tp / float(tp + fp)
        npv[i] = tn / float(tn + fn)
        sens[i] = tp / float(tp + fn)
        spec[i] = tn / float(tn + fp)
        kappa[i] = 2 * (tp*tn - fn*fp) / float((tp+fp)*(fp+tn) + (tp+fn)*(fn+tn))

    return dices, ppv, npv, sens, spec, kappa
