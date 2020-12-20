import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from python_speech_features import delta


def normalize_data(dataset, mean=None, std=None):
    if mean is None and std is None:
        d = np.vstack(dataset)
        mean = np.mean(d, axis=0)
        std = np.std(d, axis=0)

    for x in dataset:
        x -= mean
        x /= std

    return mean, std


def window_stack(x, stepsize=1, width=3):
    w2 = math.floor(width / 2.0)
    x2 = np.vstack([np.tile(x[0], (w2, 1)), x, np.tile(x[-1], (w2, 1))])  # Edges are padded with the first/last element of the sequence
    return np.hstack([x2[i:1 + i - width or None:stepsize] for i in range(0, width)])


def add_noise(dataset, noise_std):
    if isinstance(dataset, list):
        return [x + noise_std * np.random.standard_normal(size=x.shape) for x in dataset]
    else:
        return dataset + noise_std * np.random.standard_normal(size=dataset.shape)


def create_preprocessing_pipeline_sensor(win_len, pca_components):
    pipeline = []
    if win_len > 1:
        pipeline.append(('stacker', FeatureStacker(win_len)))
    if pca_components:
        pipeline.append(('mean_removal', StandardScaler(with_std=False)))
        pipeline.append(('pca', PCA(n_components=pca_components, svd_solver='full')))
    pipeline.append(('standardizer', StandardScaler()))
    pipeline.append(('dummy', DummyCustomRegressor()))
    p = Pipeline(pipeline)
    p.is_fitted_ = False
    return p


def create_preprocessing_pipeline_mfcc(mfcc_order, delta_win, acc_win):
    pipeline = []
    pipeline.append(('slicer', FeatureSlicing(np.arange(mfcc_order))))
    pipeline.append(('dynamic_params', MFCCDeltaAcc(delta_win, acc_win)))
    pipeline.append(('standardizer', StandardScaler()))
    pipeline.append(('dummy', DummyCustomRegressor()))
    p = Pipeline(pipeline)
    p.is_fitted_ = False
    return p


def preprocess_data(pipeline, X):
    dataset = X if isinstance(X, list) else [X]
    if not pipeline.is_fitted_:
        D = np.vstack(dataset)
        pipeline.fit(D, D)
        pipeline.is_fitted_ = True
    transformed_dataset = [pipeline.predict(x) for x in dataset]
    return transformed_dataset if isinstance(X, list) else transformed_dataset[0]


class FeatureStacker(TransformerMixin):
    def __init__(self, win_len=1, **kwargs):
        super().__init__(**kwargs)
        self.win_ = win_len

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X if self.win_ <= 1 else window_stack(X, stepsize=1, width=self.win_)


class FeatureSlicing(TransformerMixin):
    def __init__(self, slicing_index, **kwargs):
        super().__init__(**kwargs)
        self.idx_ = slicing_index

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X[:, self.idx_]


class MFCCDeltaAcc(TransformerMixin):
    def __init__(self, delta_win, acc_win, **kwargs):
        super().__init__(**kwargs)
        self.delta_win_ = delta_win
        self.acc_win_ = acc_win

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        delta_feats = delta(X, self.delta_win_)
        acc_feats = delta(delta_feats, self.acc_win_)
        return np.hstack([X, delta_feats, acc_feats])


class DummyCustomRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        return 0.0

    def predict(self, X):
        return X
