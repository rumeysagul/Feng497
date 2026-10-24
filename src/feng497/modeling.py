"""Karşılaştırma ve son eğitimde kullanılan ortak GPR tanımı."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paths import INPUT_COLUMNS


def make_gpr(*, log_target=False):
    """Mevcut modelin çekirdek ve optimizasyon ayarlarıyla yeni bir model kur."""
    kernel = (
        ConstantKernel(1.0)
        * Matern(length_scale=np.ones(len(INPUT_COLUMNS)), nu=2.5)
        + WhiteKernel(1e-6, (1e-10, 1e-2))
    )
    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=0,
        ),
    )
    if log_target:
        return TransformedTargetRegressor(
            regressor=model, func=np.log, inverse_func=np.exp
        )
    return model
