from typing import List, Tuple
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import skew

from configs.config_CL import config_CL


def clean_and_engineer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: DictConfig = config_CL
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Выполняет очистку пропусков, генерацию признаков из EDA и логарифмирование
    скошенных числовых признаков (порог и список колонок считаются строго по train).
    """
    id_col = cfg.data.id_column
    target_col = cfg.data.target_column

    test_ids = test_df[id_col].values
    y_raw = train_df[target_col].values
    y_train = np.log1p(y_raw).astype(np.float32) if cfg.data.log_target else y_raw.astype(np.float32)

    tr = train_df.drop(columns=[id_col, target_col], errors='ignore').copy()
    te = test_df.drop(columns=[id_col], errors='ignore').copy()

    # 1. Категории, где NaN означает отсутствие объекта (PoolQC, Alley, GarageType и т.д.)
    for col in cfg.preprocessing.none_cols:
        if col in tr.columns:
            tr[col] = tr[col].astype(object).fillna('None')
            te[col] = te[col].astype(object).fillna('None')

    # 2. Числа, где NaN означает 0 (площади/количество при отсутствии гаража или подвала)
    for col in cfg.preprocessing.zero_cols:
        if col in tr.columns:
            tr[col] = tr[col].fillna(0)
            te[col] = te[col].fillna(0)

    # 3. Оставшиеся единичные пропуски в категориях
    cat_cols = tr.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        tr[col] = tr[col].astype(object).fillna('Missing')
        te[col] = te[col].astype(object).fillna('Missing')

    # 4. Новые признаки из EDA
    if cfg.preprocessing.engineer_features:
        for df in (tr, te):
            df['HasPool'] = (df['PoolArea'] > 0).astype(np.float32)
            df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(np.float32)
            df['HasGarage'] = (df['GarageArea'] > 0).astype(np.float32)
            df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(np.float32)
            df['HasOpenPorch'] = (df['OpenPorchSF'] > 0).astype(np.float32)
            df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    num_cols = tr.select_dtypes(include=['int64', 'float64', 'float32']).columns.tolist()

    # 5. Логарифмирование сильно скошенных числовых признаков (только по train)
    if cfg.preprocessing.log_skewed_features:
        skewed_feats = tr[num_cols].apply(lambda x: skew(x.dropna()))
        high_skew = skewed_feats[abs(skewed_feats) > cfg.preprocessing.skew_threshold].index

        for feat in high_skew:
            if (tr[feat] >= 0).all() and (te[feat].dropna() >= 0).all():
                tr[feat] = np.log1p(tr[feat])
                te[feat] = np.log1p(te[feat])

    return tr, y_train, te, test_ids, num_cols, cat_cols

