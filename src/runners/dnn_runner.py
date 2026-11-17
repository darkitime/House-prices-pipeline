from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config_DNN import config_DNN
from src.data_prep.dataset import get_kfold_datasets
from src.models.tabular_nn import HousePricesModel


class EarlyStopping:
    def __init__(
        self,
        mode: str = config_DNN.earlystopping.mode,
        patience: int = config_DNN.earlystopping.patience,
        threshold: float = config_DNN.earlystopping.threshold,
        threshold_mode: str = config_DNN.earlystopping.threshold_mode
    ):
        if mode not in {'min', 'max'}:
            raise ValueError("Available values for mode: max, min")
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError("Available values for threshold_mode: rel, abs")

        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.count = 0
        self.best = None

    def changed_better(self, current: float, best: float) -> bool:
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold
        return current > best + self.threshold

    def __call__(self, tracked_parameter: float) -> bool:
        current = float(tracked_parameter)
        if self.best is None:
            self.best = current
            return False

        if self.changed_better(current, self.best):
            self.best = current
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            self.count = 0
            return True
        return False


def _resolve_device() -> str:
    if config_DNN.general.device == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = config_DNN.training.lr,
    optimizer_name: str = config_DNN.optimizer.name,
    scheduler_name: Optional[str] = config_DNN.scheduler.name,
    weight_decay: float = config_DNN.training.weight_decay,
    n_epochs: int = config_DNN.training.num_epochs,
    loss_name: str = config_DNN.loss.name,
    device: str = 'cpu',
    verbose: bool = False
) -> Dict[str, Any]:
    """Обучает одну нейросеть на одном фолде с ранней остановкой по Val RMSE."""
    model.to(device)

    loss_fn = getattr(nn, loss_name)()
    optimizer_cls = getattr(optim, optimizer_name)
    opt = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = None
    if scheduler_name in ('Plateau', 'ReduceLROnPlateau'):
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=config_DNN.scheduler.params.mode,
            factor=config_DNN.scheduler.params.factor,
            patience=config_DNN.scheduler.params.patience
        )
    elif scheduler_name in ('Cosine', 'CosineAnnealingLR'):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_epochs,
            eta_min=config_DNN.scheduler.params.eta_min
        )
    elif scheduler_name is not None:
        raise ValueError(f"Unknown scheduler_name: {scheduler_name}")

    earlystopping = EarlyStopping()
    best_val_rmse = float('inf')
    best_model_weights = deepcopy(model.state_dict())

    for epoch in range(n_epochs):
        # TRAINING
        model.train()
        train_loop = tqdm(train_loader, leave=False, disable=not verbose)
        for x_cat, x_num, targets in train_loop:
            if x_num.size(0) <= 1:
                continue

            x_cat, x_num, targets = x_cat.to(device), x_num.to(device), targets.to(device)
            pred = model(x_num, x_cat)
            loss = loss_fn(pred, targets)

            opt.zero_grad()
            loss.backward()
            if config_DNN.training.gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), config_DNN.training.clip_value)
            opt.step()

        # VALIDATION
        model.eval()
        val_se_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for x_cat, x_num, targets in val_loader:
                x_cat, x_num, targets = x_cat.to(device), x_num.to(device), targets.to(device)
                pred = model(x_num, x_cat)
                val_se_sum += ((pred - targets) ** 2).sum().item()
                val_samples += targets.size(0)

        val_rmse = float(np.sqrt(val_se_sum / val_samples))

        if lr_scheduler is not None:
            if scheduler_name in ('Plateau', 'ReduceLROnPlateau'):
                lr_scheduler.step(val_rmse)
            else:
                lr_scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_weights = deepcopy(model.state_dict())

        if earlystopping(val_rmse):
            break

    model.load_state_dict(best_model_weights)
    return {
        'model': model,
        'best_val_rmse': best_val_rmse,
        'epochs_trained': epoch + 1
    }


@torch.no_grad()
def predict_dataloader(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Возвращает предсказания в логарифмах цен для любого DataLoader."""
    model.eval()
    model.to(device)
    preds = []
    for batch in loader:
        x_cat, x_num = batch[0].to(device), batch[1].to(device)
        out = model(x_num, x_cat).cpu().numpy().ravel()
        preds.append(out)
    return np.concatenate(preds)


def run_dnn() -> Dict[str, Any]:
    """Запускает препроцессинг и 5-фолдовое обучение HousePricesModel."""
    device = _resolve_device()
    folds_data, y_train_log, _, num_cont = get_kfold_datasets()

    oof_preds_log = np.zeros(len(y_train_log), dtype=np.float64)
    n_test = len(folds_data[0]['test_dataset'])
    test_preds_log = np.zeros(n_test, dtype=np.float64)
    fold_models = []

    for fold_info in folds_data:
        fold_idx = fold_info['fold']
        train_loader = DataLoader(
            fold_info['train_dataset'],
            batch_size=config_DNN.data.train_batch_size,
            shuffle=True,
            num_workers=config_DNN.data.num_workers
        )
        val_loader = DataLoader(
            fold_info['val_dataset'],
            batch_size=config_DNN.data.val_batch_size,
            shuffle=False,
            num_workers=config_DNN.data.num_workers
        )
        test_loader = DataLoader(
            fold_info['test_dataset'],
            batch_size=config_DNN.data.test_batch_size,
            shuffle=False,
            num_workers=config_DNN.data.num_workers
        )

        emb_dims = fold_info['emb_dims'] if config_DNN.model.params.use_embeddings else None
        model = HousePricesModel(
            num_continuous_features=num_cont,
            emb_dims=emb_dims
        )

        res = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            verbose=config_DNN.general.verbose
        )
        best_model = res['model']
        fold_models.append(best_model)

        oof_preds_log[fold_info['val_idx']] = predict_dataloader(best_model, val_loader, device)
        test_preds_log += predict_dataloader(best_model, test_loader, device) / len(folds_data)

        print(f"[DNN Fold {fold_idx + 1}/{len(folds_data)}] "
              f"Best Val RMSE = {res['best_val_rmse']:.5f} (эпох: {res['epochs_trained']})")

    oof_rmse = root_mean_squared_error(y_train_log, oof_preds_log)
    print(f"[DNN] {'HousePricesModel':<20} | OOF RMSE = {oof_rmse:.5f}")

    return {
        'model_name': 'TabularNN',
        'oof_rmse': oof_rmse,
        'oof_preds_log': oof_preds_log,
        'test_preds_log': test_preds_log,
        'models': fold_models
    }