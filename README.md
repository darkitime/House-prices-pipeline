 House prices Dataset — End-to-End ML & DL Pipeline

Модульный, воспроизводимый пайплайн Classic ML и DNN для решения задачи регрессии на датасете **House Prices** (Kaggle).

Структура проекта: предобработка и конструирования признаков, кросс-валидация классических моделей (`scikit-learn`, `CatBoost`, `XGBoost`, `LightGBM`), табличной полносвязной нейросети (`PyTorch`), стекинг всех моделей.

---

## Results and leaderboard

Все модели обучались по схеме **5-Fold Stratified Cross-Validation**. Оценка качества проводилась на валидационных фолдах (Out-Of-Fold Accuracy) и на платформе Kaggle (Public Leaderboard Score).

| Модель | OOF RMSE | Public Score (PS) |
| :--- | :---: | :---: |
| **Ridge** | 0.13558 | 0.13466 |
| **TabularMLP** | 0.16672 | 0.15090 |
| **Lasso** | 0.13198 | 0.12721 |
| **ElasticNet** | 0.13959 | 0.13693 |
| **SVR** | 0.12999 | 0.12925 |
| **CatBoostRegressor** | **0.12237** | 0.12449 |
| **LGBMRegressor** | 0.12757 | 0.12274 |
| **XGBRegressor** | 0.12543 | 0.12648 |
| **StackingRegressor** | 0.12238 | **0.12087** |

### Ключевые выводы
* **Обобщающая способность:**  `LGBMRegressor` и `StackingRegressor` показали наилучшие результаты на PS (**0.12274** и **0.12087** соответственно). На валидации лидирует `CatBoostRegressor` (**0.12237** ). Скорее всего кетбуст переобучился, запомнив некоторую сложную природу тренировочной выборки, которой потом не было в тестовой.
* **Ансамблирование:** Стекинг показал лучший результат на PS


---

## Архитектура проекта

```text
├── configs/                     # Конфигурации экспериментов
│   ├── config_CL.py             # Настройки классических моделей
│   └── config_DNN.py            # Настройки нейросети
│
├── data/                        # Директория с данными
│   ├── train.csv
│   ├── test.csv
│   └── real_test_submission.csv # Разметка для теста (получена в notebooks/eda.ipynb)
│
├── src/
│   ├── data_prep/
│   │   ├── preprocessing.py     # Очистка, заполнение пропусков, OHE
│   │   └── dataset.py           # Stratified K-Fold PyTorch Dataset
│   ├── models/
│   │   ├── classic_models.py    # Scikit-Learn / Boostings + OOF train
│   │   └── tabular_nn.py        # Архитектура TabularMLP (Embeddings + Residual blocks)
│   ├── engine/
│   │   ├── classic_runner.py    # Пайплайн запуска классического ML
│   │   └── dnn_runner.py        # Тренировочный цикл PyTorch, валидация и EarlyStopping
│   └── utils.py                 # Фиксация сидов, сохранение чекпоинтов и лидерборда
│
├── checkpoints/                 # Сохраненные веса (.joblib и .pt, в .gitignore)
├── submissions/                 # Готовые сабмиты и leaderboard.csv (в .gitignore)
│
├── main.py                      # Обучение и оценка всех моделей
```
### Замечания

Для улучшения архитектуры стоит добавить переход из train mode в validation mode для поиска гиперпараметров.

## Инструкция по установке и запуску

Склонируйте репозиторий в свое рабочее пространство. Настройте окружение (файл requirements.txt). Обучите все модели и получите лидерборд (python main.py). Итоговые результаты будут сохранены в папку submissions/
