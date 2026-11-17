from omegaconf import OmegaConf

_config_dnn = {
    'general': {
        'mode': 'train',
        'experiment_name': 'house_prices_tabular_nn_v1',
        'seed': 42,
        'device': 'cuda',  # Автоматически переключится на 'cpu', если GPU недоступен
        'verbose': False
    },
    'paths': {
        'train': 'data/train.csv',
        'test': 'data/test.csv',
        'targets': None,
        'checkpoints_dir': './checkpoints/dnn',
        'submission_dir': './submissions',
        'log_name': 'dnn_train.log',
    },
    'data': {
        'id_column': 'Id',
        'target_column': 'SalePrice',
        'log_target': True,
        'n_splits': 5,
        'batch_size': 32,
        'train_batch_size': '${data.batch_size}',
        'val_batch_size': 64,
        'test_batch_size': 64,
        'num_workers': 0,
        'max_emb_dim': 50,
    },
    'model': {
        'name': 'HousePricesModel',
        'params': {
            'use_embeddings': True,
            'hidden_dim': 32,
            'num_blocks': 3,
            'activation_cls': 'SiLU',
            'use_batchnorm': False,
            'dropout_p': 0.2,
            'output_dim': 1,
        }
    },
    'training': {
        'num_epochs': 120,
        'lr': 0.003,
        'weight_decay': 0.0001,
        'gradient_clipping': True,
        'clip_value': 1.0,
    },
    'optimizer': {
        'name': 'AdamW',
        'params': {
            'lr': '${training.lr}',
            'weight_decay': '${training.weight_decay}',
        }
    },
    'scheduler': {
        'name': 'CosineAnnealingLR',
        'params': {
            'eta_min': 1e-6,
            'mode': 'min',
            'factor': 0.5,
            'patience': 4
        }
    },
    'earlystopping': {
        'mode': 'min',
        'patience': 12,
        'threshold': 1e-4,
        'threshold_mode': 'rel'
    },
    'loss': {
        'name': 'MSELoss',
    },
    'metric': {
        'name': 'rmse',
    }
}

config_DNN = OmegaConf.create(_config_dnn)