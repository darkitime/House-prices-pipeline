from typing import List, Optional, Tuple, Type
import torch
import torch.nn as nn

from configs.config_DNN import config_DNN


class TabularBlock(nn.Module):
    """Linear -> [BatchNorm1d] -> Activation -> [Dropout]"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_cls: Type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = True,
        dropout_p: float = 0.0
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))

        layers.append(activation_cls())

        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HousePricesModel(nn.Module):
    def __init__(
        self,
        num_continuous_features: int,
        emb_dims: Optional[List[Tuple[int, int]]] = None,
        hidden_dim: int = config_DNN.model.params.hidden_dim,
        num_blocks: int = config_DNN.model.params.num_blocks,
        activation_cls: Type[nn.Module] = getattr(nn, config_DNN.model.params.activation_cls),
        use_batchnorm: bool = config_DNN.model.params.use_batchnorm,
        dropout_p: float = config_DNN.model.params.dropout_p,
        output_dim: int = config_DNN.model.params.output_dim
    ):
        super().__init__()
        self.use_embeddings = emb_dims is not None and len(emb_dims) > 0

        if self.use_embeddings:
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in emb_dims
            ])
            total_emb_dim = sum(emb_dim for _, emb_dim in emb_dims)
        else:
            self.embeddings = None
            total_emb_dim = 0

        in_dim = num_continuous_features + total_emb_dim

        blocks = []
        for i in range(num_blocks):
            current_in = in_dim if i == 0 else hidden_dim
            blocks.append(
                TabularBlock(
                    in_features=current_in,
                    out_features=hidden_dim,
                    activation_cls=activation_cls,
                    use_batchnorm=use_batchnorm,
                    dropout_p=dropout_p
                )
            )
        self.feature_extractor = nn.Sequential(*blocks)

        hidden_in = hidden_dim if num_blocks > 0 else in_dim
        self.regressor = nn.Linear(hidden_in, output_dim)

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_embeddings and x_cat is not None:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_emb = torch.cat(emb_outs, dim=1)
            x = torch.cat([x_num, x_emb], dim=1)
        else:
            x = x_num

        features = self.feature_extractor(x)
        return self.regressor(features)