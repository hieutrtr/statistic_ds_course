import numpy as np
from sklearn.model_selection import KFold

def select_bin(dataset, fields):
    """
    This function applies cross-validation to identify the best bin for the given dataset and fields.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset containing the data.
    fields : list
        List of field names in the dataset for which the bin selection should be performed.

    Returns
    -------
    best_bins : dict
        A dictionary containing the best bin selection for each field.
    """
    best_bins = {}
    for field in fields:
        data = dataset[field].values
        best_score = 0
        best_bin = 0

        for bins in range(2, 10):
            scores = []
            kf = KFold(n_splits=10, shuffle=True)

            for train_index, test_index in kf.split(data):
                X_train, X_test = data[train_index], data[test_index]
                q75, q25 = np.percentile(X_train, [75, 25])
                iqr = q75 - q25
                bin_width = 2 * (iqr / len(X_train) ** (1 / 3))
                num_bins = int((max(X_train) - min(X_train)) / bin_width)

                if num_bins < bins:
                    continue

                hist, _ = np.histogram(X_train, bins=num_bins)
                hist_norm = hist / len(X_train)

                bin_edges = np.linspace(min(X_train), max(X_train), num_bins + 1)
                bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

                cdf = np.diff(np.percentile(X_train, np.linspace(0, 100, num_bins + 1)))
                score = np.sum((hist_norm - cdf) ** 2)

                scores.append(score)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_bin = bins

        best_bins[field] = best_bin

    return best_bins