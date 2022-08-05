import numpy as np
import cv2

from numpy import argmax
from sklearn.metrics import roc_curve

def best_thresh(y_true, y_pred):
    """
    
    """

    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_true.flatten(), y_pred.flatten())

    gmeans = np.sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    best_thresh = thresholds[ix]

    print('Best Threshold=%f, G-Mean=%.3f' % (best_thresh, gmeans[ix]))

    # # Youden's J statistic
    # J = tpr - fpr
    # ix = argmax(J)
    # best_thresh = thresholds[ix]
    # print('Best Threshold=%f' % (best_thresh))

    return best_thresh

def binary_masks(groundtruth, predictions):
    """
    
    """

    assert groundtruth.shape[0:3] == predictions.shape[0:3], "Groundtruth and prediction shapes don't match"

    binary_masks = []

    for i in range(groundtruth.shape[0]):
        label = groundtruth[i, :, :].astype(int)

        if not np.any(label): 
            binary_masks.append(label)
            continue

        out_mask = predictions[i, :, :]

        thresh = best_thresh(label, out_mask)

        ret, mask = cv2.threshold(out_mask, thresh, 1, cv2.THRESH_BINARY)
        binary_masks.append(mask)

    return np.array(binary_masks)