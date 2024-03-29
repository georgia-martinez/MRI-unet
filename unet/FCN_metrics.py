import keras.backend as K
import numpy as np
import tensorflow as tf

from sklearn import metrics
from scipy.spatial.distance import directed_hausdorff

def dice_coef(y_true, y_pred, smooth=1.0):
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

    y_true = K.cast_to_floatx(y_true)

    flat_truth = K.flatten(y_true)
    flat_prediction = K.flatten(y_pred)
    intersection = K.sum(flat_truth * flat_prediction)
    numerator = 2 * intersection + smooth
    sum_truth = K.sum(flat_truth)
    sum_prediction = K.sum(flat_prediction)
    denominator = sum_truth + sum_prediction + smooth
    dice = numerator / denominator
    
    return dice # <class 'tensorflow.python.framework.ops.Tensor'>

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

# def hausdorff(y_true, y_pred):

#     haus = directed_hausdorff(y_true, y_pred)[0]
#     haus = tf.convert_to_tensor(haus)

#     print(type(haus))

#     return haus

def hausdorff(A, B):
    """
    Original code: https://github.com/danielenricocahall/Keras-Weighted-Hausdorff-Distance-Loss/blob/master/hausdorff/hausdorff.py

    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """

    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D

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

