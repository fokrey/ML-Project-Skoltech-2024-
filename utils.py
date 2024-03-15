from sklearn.metrics import precision_score, recall_score, matthews_corrcoef
import numpy as np
from itertools import combinations
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def f_measure(y_true, y_pred, beta=1):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f_measure = (1 + beta) * (precision * recall) / (beta * precision + recall)
    return f_measure

def mmcc(y_true, y_pred, classes):
    mcc_values = []
    for class_pair in combinations(classes, 2):
        # Create binary arrays for each class in the pair
        y_true_binary = np.isin(y_true, class_pair)
        y_pred_binary = np.isin(y_pred, class_pair)
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        mcc_values.append(mcc)
    mmcc_value = np.mean(mcc_values)
    return mmcc_value


def macro_averaged_auprc(y_true, y_scores, n_classes):
    # Ensure y_true is binarized for multi-class labels
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Initialize list to store AUC scores for each class
    auc_scores = []
    
    # Compute Precision-Recall curve and AUC for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        auc_score = auc(recall, precision)
        auc_scores.append(auc_score)
    
    # Calculate the macro-averaged AUPRC
    macro_auprc = np.mean(auc_scores)
    return macro_auprc
def g_mean_multiclass(y_true, y_pred, n_classes):
    """
    Compute the G-mean for multi-class classification.
    
    Parameters:
    - y_true: array-like of shape (n_samples,), True labels for each sample.
    - y_pred: array-like of shape (n_samples,), Predicted labels for each sample.
    - n_classes: int, The number of unique classes.
    
    Returns:
    - g_mean: The geometric mean of recall for all classes.
    """
    recalls = []
    for i in range(n_classes):
        recall = recall_score(y_true, y_pred, labels=[i], average='weighted')
        recalls.append(recall)
    g_mean = np.sqrt(np.prod(recalls))
    return g_mean,recalls