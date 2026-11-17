from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, TargetEncoder
from sklearn.svm import SVR

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

MODEL_REGISTRY = {
    'Ridge': Ridge,
    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    'SVR': SVR,
    'CatBoostRegressor': CatBoostRegressor,
    'LGBMRegressor': LGBMRegressor,
    'XGBRegressor': XGBRegressor,
}

SCALER_REGISTRY = {
    'StandardScaler': StandardScaler,
    'RobustScaler': RobustScaler,
}


def build_classic_pipeline(
    model_name: str,
    cfg: DictConfig,
    num_cols: List[str],
    cat_cols: List[str]
) -> Pipeline:
    """Создаёт полный sklearn Pipeline (Imputer + Scaler + TargetEncoder + Model) из конфига."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Модель '{model_name}' не найдена. Доступны: {list(MODEL_REGISTRY.keys())}")
    if model_name not in cfg.models:
        raise KeyError(f"Конфигурация для '{model_name}' не найдена в cfg.models")

    model_cfg = OmegaConf.to_container(cfg.models[model_name], resolve=True)
    prep_params = model_cfg.get('prep_params', {})
    model_params = model_cfg.get('model_params', {})

    scaler_name = prep_params.get('scaler', 'StandardScaler')
    scaler_cls = SCALER_REGISTRY[scaler_name]
    encoder_smooth = prep_params.get('encoder_smooth', 'auto')

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler_cls())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', TargetEncoder(
            target_type='continuous',
            smooth=encoder_smooth,
            cv=5,
            random_state=cfg.general.seed
        ))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    model_cls = MODEL_REGISTRY[model_name]
    return Pipeline([
        ('prep', preprocessor),
        ('model', model_cls(**model_params))
    ])


def train_and_eval_oof(
    model_name: str,
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_test: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str]
) -> Dict[str, Any]:
    """Обучает пайплайн на KFold, вычисляет OOF RMSE и усредняет предсказания на тесте."""
    kf = KFold(
        n_splits=cfg.training.n_splits,
        shuffle=True,
        random_state=cfg.general.seed
    )

    oof_preds_log = np.zeros(len(X_train), dtype=np.float64)
    test_preds_log = np.zeros(len(X_test), dtype=np.float64)
    fold_models = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train_log[train_idx]
        X_va = X_train.iloc[val_idx]

        pipe = build_classic_pipeline(model_name, cfg, num_cols, cat_cols)
        pipe.fit(X_tr, y_tr)

        oof_preds_log[val_idx] = pipe.predict(X_va)
        test_preds_log += pipe.predict(X_test) / cfg.training.n_splits
        fold_models.append(pipe)

    oof_rmse = root_mean_squared_error(y_train_log, oof_preds_log)

    return {
        'model_name': model_name,
        'oof_rmse': oof_rmse,
        'oof_preds_log': oof_preds_log,
        'test_preds_log': test_preds_log,
        'models': fold_models
    }