import numpy as np
from sklearn.model_selection import KFold

def cross_validation(X):
    """
    This function implements a cross-validation approach to identify the best bin
    for a given dataset.
  
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
  
    Returns
    -------
    bins : int
        Optimal number of bins for the given dataset
    """
    best_score = 0
    best_bin = 0
    for bins in range(2, 10):
        scores = []
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            # Calculate the bin width
            bin_width = (max(X_train) - min(X_train)) / bins
            # Bin the data
            binned_X_train = np.floor((X_train - min(X_train)) / bin_width)
            binned_X_test = np.floor((X_test - min(X_train)) / bin_width)
            # Calculate score
            score = np.mean(binned_X_train == binned_X_test)
            scores.append(score)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_bin = bins
    return best_bin

def select_bin(dataset, fields):
    bins = {}
    for field in fields:
        x = dataset[field].values
        x_binned = cross_validation(x)
        bins[field] = x_binned
    return bins