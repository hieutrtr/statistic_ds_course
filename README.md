## Quickstart
```
python3 -m venv .venv
source .venv/bin/activate
```

```
pip3 install -r requirements.txt
```

```
python3 app.py
```

## Bin selection methods

See methods implementation in `methods` directory.

#### Cross validation
```python
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
```

#### Scott
```python
def Scott_bins(data):
    # Calculate the interquartile range.
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25

    # Calculate the optimal bin width.
    bin_width = 2 * (iqr/len(data)**(1/3))

    # Calculate the number of bins.
    num_bins = int((max(data) - min(data))/bin_width)

    # Return the optimal bin width and number of bins.
    return bin_width, num_bins
```

#### Sturges
```python
def sturges_rule(data):
    '''
    Function to calculate the number of bins to be used in a histogram using Sturges' Rule.
    data: Array of numerical data
    '''
    num_bins = 1 + np.log2(len(data))
    return num_bins
```