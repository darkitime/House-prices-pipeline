from omegaconf import OmegaConf

_config_cl = {
    'general': {
        'seed': 42,
        'mode': 'train',  # 'train', 'inference'
        'metric': 'rmse',
    },
    'paths': {
        'train': 'data/train.csv',
        'test': 'data/test.csv',
        'targets': None,  # Опционально: путь к файлу с истинными ценами теста для локальной проверки
        'output_dir': './outputs/classic_ml',
        'submission_dir': './submissions',
        'weights_dir': './checkpoints/classic_ml',
        'log_name': 'classic_ml.log',
    },
    'data': {
        'id_column': 'Id',
        'target_column': 'SalePrice',
        'log_target': True,
        'drop_columns': ['Id'],
    },
    'preprocessing': {
        'engineer_features': True,
        'log_skewed_features': True,
        'skew_threshold': 0.75,
        'none_cols': [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
        ],
        'zero_cols': [
            'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'
        ],
    },
    'training': {
        'n_splits': 5,
        'use_stacking': True,
        'stacking_estimators': [
            'CatBoostRegressor',
            'SVR',
            'LGBMRegressor',
            'XGBRegressor',
            'Ridge',
            'Lasso'
        ],
    },
    'models': {
        'Ridge': {
            'prep_params': {'scaler': 'RobustScaler', 'encoder_smooth': 5.0},
            'model_params': {
                'alpha': 5.0,
                'positive': True,
                'random_state': '${general.seed}'
            }
        },
        'Lasso': {
            'prep_params': {'scaler': 'RobustScaler', 'encoder_smooth': 5.0},
            'model_params': {
                'alpha': 0.00035,
                'positive': True,
                'selection': 'random',
                'max_iter': 10000,
                'random_state': '${general.seed}'
            }
        },
        'ElasticNet': {
            'prep_params': {'scaler': 'StandardScaler', 'encoder_smooth': 'auto'},
            'model_params': {
                'alpha': 0.07,
                'l1_ratio': 0.05,
                'selection': 'cyclic',
                'max_iter': 10000,
                'random_state': '${general.seed}'
            }
        },
        'SVR': {
            'prep_params': {'scaler': 'StandardScaler', 'encoder_smooth': 'auto'},
            'model_params': {
                'C': 8.0,
                'epsilon': 0.01,
                'gamma': 'scale',
                'kernel': 'rbf'
            }
        },
        'CatBoostRegressor': {
            'prep_params': {'scaler': 'StandardScaler', 'encoder_smooth': 'auto'},
            'model_params': {
                'bootstrap_type': 'Bernoulli',
                'depth': 5,
                'iterations': 1600,
                'l2_leaf_reg': 2,
                'learning_rate': 0.035,
                'rsm': 0.9,
                'subsample': 0.9,
                'random_seed': '${general.seed}',
                'verbose': 0,
                'thread_count': -1
            }
        },
        'LGBMRegressor': {
            'prep_params': {'scaler': 'StandardScaler', 'encoder_smooth': 'auto'},
            'model_params': {
                'extra_trees': True,
                'n_estimators': 300,
                'learning_rate': 0.05,
                'min_child_samples': 10,
                'num_leaves': 31,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'subsample': 0.75,
                'random_state': '${general.seed}',
                'verbose': -1,
                'n_jobs': -1
            }
        },
        'XGBRegressor': {
            'prep_params': {'scaler': 'StandardScaler', 'encoder_smooth': 'auto'},
            'model_params': {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'colsample_bytree': 0.7,
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 5,
                'reg_alpha': 0.0,
                'reg_lambda': 2.0,
                'subsample': 0.7,
                'random_state': '${general.seed}',
                'n_jobs': -1
            }
        }
    },
    'to_train': [
        'Ridge',
        'Lasso',
        'ElasticNet',
        'SVR',
        'CatBoostRegressor',
        'LGBMRegressor',
        'XGBRegressor'
    ]
}

config_CL = OmegaConf.create(_config_cl)