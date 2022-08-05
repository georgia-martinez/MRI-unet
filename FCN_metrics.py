import keras.backend as K
import numpy as np

from sklearn import metrics

def dice_coef(y_true, y_pred, smooth = 1.0):
    """Compute Dice Similarity Coefficient (DSC)
        
        Parameters
        ----------
        y_true : 1D numpy array of float32
        Array of correct/ground truth labels
        y_pred : 1D numpy array of predictions
        Array of predicted labels generated by FCN
        smooth : float
        Defaults to 1.0
        
        Returns
        -------
        Returns the DSC
        
        Additional Notes
        -------
        Original source code can be found `here <https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py>`_
        """

    # y_true = K.cast_to_floatx(y_true)

    flat_truth = K.flatten(y_true)
    flat_prediction = K.flatten(y_pred)
    intersection = K.sum(flat_truth * flat_prediction)
    numerator = 2 * intersection + smooth
    sum_truth = K.sum(flat_truth)
    sum_prediction = K.sum(flat_prediction)
    denominator = sum_truth + sum_prediction + smooth
    dice = numerator / denominator
    return dice


def dice_coef_loss(y_true, y_pred):
    """Dice Similarity Coefficient (DSC) loss function
    
    Parameters
    ----------
    y_true : 1D numpy array of float32
    Array of correct/ground truth labels
    y_pred : 1D numpy array of predictions
    Array of predicted labels generated by FCN
    
    Returns
    -------
    Returns the DSC loss
    
    """
    return -dice_coef(y_true, y_pred)


def compute_dsc(preds, gt):
    """Compute average Dice Similarity Coefficient (DSC) given predictions and ground truths
    
    Parameters
        ----------
        y_true : 1D numpy array of float32
        Array of correct/ground truth labels
        y_pred : 1D numpy array of predictions
        Array of predicted labels generated by FCN
        smooth : float
        Defaults to 1.0
        
        Returns
        -------
        Returns the average DSC
    """
    dice_list = []
    for ind in range(len(gt)):
        seg = preds[ind]
        actual = gt[ind]
        dice = dice_coef(actual, seg)
        dice_list.append(dice)
    dice_list = np.array(dice_list)
    dice_avg = np.average(dice_list)
    return dice_avg


def compute_acc(preds, gt):
    """Compute average accuracy given predictions and ground truths
    
        Parameters
        ----------
        y_true : 1D numpy array of float32
        Array of correct/ground truth labels
        y_pred : 1D numpy array of predictions
        Array of predicted labels generated by FCN
        smooth : float
        Defaults to 1.0
        
        Returns
        -------
        Returns the average accuracy
    """
    accuracy_list = []
    for ind in range(len(gt)):
        seg = preds[ind]
        actual = gt[ind]
        acc = metrics.accuracy_score(actual.squeeze(),seg.squeeze())
        accuracy_list.append(acc)
    accuracy_list = np.array(accuracy_list)
    acc_avg = np.average(accuracy_list)
    return acc_avg

