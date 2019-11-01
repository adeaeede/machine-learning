import numpy as np
import pandas as pd



class KNNClassifier:
    def __init__(self, train_x, train_y):
        self._train_x = train_x
        self._train_y = train_y

    def classify(self, X, k):
        def classify_single(x):
            distances = np.sqrt(np.sum(np.square(self._train_x - x), axis=1))
            nearest_labels = self._train_y[np.argpartition(distances, k)[:k]]
            predicted = np.argmax(np.bincount(nearest_labels))
            return predicted
        if X.ndim == 1:
            return classify_single(X)
        elif X.ndim == 2:
            y_pred = np.empty(len(X), dtype=np.uint8)
            for i in range(len(X)):
                y_pred[i] = classify_single(X[i])
            return y_pred
        else:
            raise ValueError('invalid input shape')


if __name__ == '__main__':
    _training_data = np.array(pd.read_csv('zip.train', sep=' ', header=None), dtype=np.float32)
    _test_data = np.array(pd.read_csv('zip.test', sep=' ', header=None), dtype=np.float32)

    _train_x = _training_data[:, 1:-1]
    _train_y = _training_data[:, 0].astype(np.uint8)

    _test_x = _test_data[:, 1:]
    _test_y = _test_data[:, 0].astype(np.uint8)

    for k in range(1, 10):
        _n_test_samples = 100
        _knn = KNNClassifier(_train_x, _train_y)
        _y_pred = _knn.classify(_test_x[:_n_test_samples], k)
        _accuracy = np.sum(np.equal(_y_pred, _test_y[:_n_test_samples])) / len(_y_pred)
        print(_accuracy)
