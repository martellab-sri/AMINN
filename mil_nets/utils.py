import numpy as np


def convertToBatch(bags):
    data_set = []
    for ibag, bag in enumerate(bags):
        batch_data = np.asarray(bag[0], dtype='float32')
        batch_label = np.asarray(bag[1])
        batch_id = np.asarray(bag[2], dtype='int16')
        data_set.append((batch_data, batch_label, batch_id))
    return data_set


def feature_sets(df, feature_class='original', normalize=True):
    # Keep original features only and discard Laplacian of Gaussian (Log) features and wavelet features
    idxs = df.columns.str.contains(r'sigma|wavelet')
    if feature_class == 'original':
        df = df.iloc[:, ~idxs]
    # Hard-coded feature columns, should be changed based on specific datasets
    X = df.iloc[:, 15: -5]
    # Two step normalization
    if normalize:
        # Log transformation with a correction term
        X = X - X.min() + np.median(X - X.min())
        X = np.log(X)
    X = (X - X.mean()) / X.std()
    # Column 0: Patient ID, column -2: follow up time and column -1, mortality were passed as bag labels
    Y = df.iloc[:, [0, -2, -1]]
    return X, Y
