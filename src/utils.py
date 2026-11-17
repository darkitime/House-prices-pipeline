import os
import random
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error

from configs.config_CL import config_CL
from configs.config_DNN import config_DNN


def set_seed(seed: int = config_CL.general.seed) -> None:
    """Фиксация всех генераторов случайных чисел для воспроизводимости."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(
    ids: np.ndarray,
    preds_log: np.ndarray,
    filename: str,
    output_dir: str = config_CL.paths.submission_dir,
    inverse_log: bool = True
) -> str:
    """Переводит предсказания из логарифма в доллары (np.expm1) и сохраняет CSV"""
    os.makedirs(output_dir, exist_ok=True)
    final_preds = np.expm1(preds_log) if inverse_log else preds_log

    sub = pd.DataFrame({
        config_CL.data.id_column: ids,
        config_CL.data.target_column: final_preds
    })

    filepath = os.path.join(output_dir, filename)
    sub.to_csv(filepath, index=False)
    return filepath


def build_leaderboard(
    all_results: Dict[str, Dict[str, Any]],
    y_test_true_log: Optional[np.ndarray] = None,
    output_dir: str = config_CL.paths.submission_dir
) -> pd.DataFrame:
    """Формирует сводную таблицу моделей, сортирует по OOF_RMSE (меньше = лучше) и сохраняет CSV."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for model_name, res in all_results.items():
        row = {
            'Model': model_name,
            'OOF_RMSE': round(float(res['oof_rmse']), 5),
        }
        if y_test_true_log is not None and 'test_preds_log' in res:
            test_rmse = root_mean_squared_error(y_test_true_log, res['test_preds_log'])
            row['Test_RMSE'] = round(float(test_rmse), 5)
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    sort_col = 'Test_RMSE' if 'Test_RMSE' in leaderboard.columns else 'OOF_RMSE'
    leaderboard = leaderboard.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

    leaderboard_path = os.path.join(output_dir, 'leaderboard.csv')
    leaderboard.to_csv(leaderboard_path, index=False)
    return leaderboard


def save_classic_checkpoints(
    results: Dict[str, Dict[str, Any]],
    weights_dir: str = config_CL.paths.weights_dir
) -> None:
    """Сохраняет обученные фолдовые пайплайны классических моделей."""
    os.makedirs(weights_dir, exist_ok=True)
    for model_name, res in results.items():
        if 'models' in res:
            filepath = os.path.join(weights_dir, f"{model_name}_folds.joblib")
            joblib.dump(res['models'], filepath)


def save_dnn_checkpoints(
    models: List[torch.nn.Module],
    checkpoints_dir: str = config_DNN.paths.checkpoints_dir
) -> None:
    """Сохраняет веса каждого фолда нейросети."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    for fold_idx, model in enumerate(models):
        filepath = os.path.join(checkpoints_dir, f"dnn_fold_{fold_idx}.pt")
        torch.save(model.state_dict(), filepath)