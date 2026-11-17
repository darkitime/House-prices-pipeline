from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from configs.config_CL import config_CL
from src.data_prep.preprocessing import clean_and_engineer
from src.models.classic_models import build_classic_pipeline, train_and_eval_oof


def run_cl() -> Tuple[Dict[str, Dict[str, Any]], np.ndarray]:
    """Запускает препроцессинг, KFold обучение всех классических моделей и StackingRegressor."""
    train_df = pd.read_csv(config_CL.paths.train)
    test_df = pd.read_csv(config_CL.paths.test)

    X_train, y_train_log, X_test, test_ids, num_cols, cat_cols = clean_and_engineer(
        train_df, test_df, cfg=config_CL
    )

    results: Dict[str, Dict[str, Any]] = {}

    for model_name in config_CL.to_train:
        res = train_and_eval_oof(
            model_name=model_name,
            cfg=config_CL,
            X_train=X_train,
            y_train_log=y_train_log,
            X_test=X_test,
            num_cols=num_cols,
            cat_cols=cat_cols
        )
        results[model_name] = res
        print(f"[Classic] {model_name:<20} | OOF RMSE = {res['oof_rmse']:.5f}")

    # Сборка StackingRegressor поверх лучших моделей из конфига
    if config_CL.training.use_stacking:
        print("\n[Classic] Обучение StackingRegressor...")
        estimators_list = [
            (name, build_classic_pipeline(name, config_CL, num_cols, cat_cols))
            for name in config_CL.training.stacking_estimators
            if name in config_CL.models
        ]

        kf = KFold(
            n_splits=config_CL.training.n_splits,
            shuffle=True,
            random_state=config_CL.general.seed
        )

        stacking_model = StackingRegressor(
            estimators=estimators_list,
            final_estimator=RidgeCV(alphas=np.logspace(-2, 2, 50)),
            cv=kf,
            n_jobs=-1,
            passthrough=False
        )

        # Честный OOF прогноз стекинга для сравнения в лидерборде
        stack_oof_log = cross_val_predict(
            stacking_model,
            X_train,
            y_train_log,
            cv=kf,
            n_jobs=-1
        )
        stack_oof_rmse = root_mean_squared_error(y_train_log, stack_oof_log)

        # Обучаем финальный стекинг на всём X_train для прогноза на тесте
        stacking_model.fit(X_train, y_train_log)
        stack_test_log = stacking_model.predict(X_test)

        results['StackingRegressor'] = {
            'model_name': 'StackingRegressor',
            'oof_rmse': stack_oof_rmse,
            'oof_preds_log': stack_oof_log,
            'test_preds_log': stack_test_log,
            'models': [stacking_model]
        }
        print(f"[Classic] {'StackingRegressor':<20} | OOF RMSE = {stack_oof_rmse:.5f}")

    return results, test_ids