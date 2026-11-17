import os
import pandas as pd

from configs.config_CL import config_CL
from src.runners.classic_runner import run_cl
from src.runners.dnn_runner import run_dnn
from src.utils import (
    build_leaderboard,
    save_classic_checkpoints,
    save_dnn_checkpoints,
    save_submission,
    set_seed,
)


def main():
    set_seed(config_CL.general.seed)
    print("=== Запуск пайплайна House Prices ===\n")

    # 1. Обучение классических моделей и StackingRegressor
    classic_results, test_ids = run_cl()
    save_classic_checkpoints(classic_results)

    # 2. Обучение Tabular DNN на 5 фолдах
    print("\n=== Обучение Tabular DNN ===")
    dnn_result = run_dnn()
    save_dnn_checkpoints(dnn_result['models'])

    # 3. Объединение всех результатов и построение лидерборда
    all_results = {**classic_results, 'TabularNN': dnn_result}

    y_test_true_log = None
    if config_CL.paths.targets and os.path.exists(config_CL.paths.targets):
        y_test_raw = pd.read_csv(config_CL.paths.targets)[config_CL.data.target_column].values
        import numpy as np
        y_test_true_log = np.log1p(y_test_raw)

    leaderboard = build_leaderboard(all_results, y_test_true_log=y_test_true_log)

    print("\n================ ИТОГОВЫЙ ЛИДЕРБОРД ================")
    print(leaderboard.to_string(index=False))
    print("====================================================")

    # 4. Сохранение сабмитов для всех моделей и отдельного файла для лучшей модели
    for model_name, res in all_results.items():
        save_submission(
            ids=test_ids,
            preds_log=res['test_preds_log'],
            filename=f"submission_{model_name.lower()}.csv"
        )

    best_model_name = leaderboard.loc[0, 'Model']
    best_path = save_submission(
        ids=test_ids,
        preds_log=all_results[best_model_name]['test_preds_log'],
        filename="submission_best.csv"
    )
    print(f"\nЛучшая модель по CV: {best_model_name} (OOF RMSE = {leaderboard.loc[0, 'OOF_RMSE']:.5f})")
    print(f"Главный сабмит сохранён в: {best_path}")


if __name__ == '__main__':
    main()