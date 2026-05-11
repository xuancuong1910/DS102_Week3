import numpy as np

def evaluate_metrics(y_true, y_pred):
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1)) 
    fp = np.sum((y_true == -1) & (y_pred == 1))  
    fn = np.sum((y_true == 1) & (y_pred == -1))  

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1