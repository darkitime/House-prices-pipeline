from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from configs.config_CL import config_CL
from configs.config_DNN import config_DNN
from src.data_prep.preprocessing import clean_and_engineer


class HousePricesDataset(Dataset):
    """PyTorch Dataset для табличных данных (категории + числа)."""
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: Optional[np.ndarray] = None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X_num)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]


class TabularNNPreprocessor:
    """Масштабирует числа и кодирует категории в индексы 1..N (0 = <UNK>) внутри фолда."""
    def __init__(self, num_cols: List[str], cat_cols: List[str], max_emb_dim: int = 50):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.max_emb_dim = max_emb_dim

        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.emb_dims: List[Tuple[int, int]] = []

    def fit(self, X_train: pd.DataFrame) -> 'TabularNNPreprocessor':
        num_imputed = self.imputer.fit_transform(X_train[self.num_cols])
        self.scaler.fit(num_imputed)

        self.cat_mappings = {}
        self.emb_dims = []

        for col in self.cat_cols:
            unique_vals = X_train[col].astype(str).unique()
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.cat_mappings[col] = mapping

            num_classes = len(unique_vals) + 1
            emb_dim = min(self.max_emb_dim, (num_classes + 1) // 2)
            self.emb_dims.append((num_classes, emb_dim))

        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        num_imputed = self.imputer.transform(X[self.num_cols])
        X_num = self.scaler.transform(num_imputed).astype(np.float32)

        X_cat = np.zeros((len(X), len(self.cat_cols)), dtype=np.int64)
        for i, col in enumerate(self.cat_cols):
            mapping = self.cat_mappings[col]
            X_cat[:, i] = X[col].astype(str).map(mapping).fillna(0).astype(np.int64).values

        return X_cat, X_num

    def fit_transform(self, X_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(X_train).transform(X_train)


def get_kfold_datasets(
    path_to_train: str = config_DNN.paths.train,
    path_to_test: str = config_DNN.paths.test,
    n_splits: int = config_DNN.data.n_splits,
    seed: int = config_DNN.general.seed,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, int]:
    """Разбивает данные на KFold и готовит изолированные датасеты без утечки данных."""
    train_df = pd.read_csv(path_to_train)
    test_df = pd.read_csv(path_to_test)

    X_train, y_train_log, X_test, test_ids, num_cols, cat_cols = clean_and_engineer(
        train_df, test_df, cfg=config_CL
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr_f = X_train.iloc[train_idx]
        y_tr_f = y_train_log[train_idx]
        X_va_f = X_train.iloc[val_idx]
        y_va_f = y_train_log[val_idx]

        prep = TabularNNPreprocessor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            max_emb_dim=config_DNN.data.max_emb_dim
        )
        X_cat_tr, X_num_tr = prep.fit_transform(X_tr_f)
        X_cat_va, X_num_va = prep.transform(X_va_f)
        X_cat_te, X_num_te = prep.transform(X_test)

        folds_data.append({
            'fold': fold_idx,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'emb_dims': prep.emb_dims,
            'train_dataset': HousePricesDataset(X_cat_tr, X_num_tr, y_tr_f),
            'val_dataset': HousePricesDataset(X_cat_va, X_num_va, y_va_f),
            'test_dataset': HousePricesDataset(X_cat_te, X_num_te),
        })

    return folds_data, y_train_log, test_ids, len(num_cols)
