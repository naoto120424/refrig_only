import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os, math

from copy import deepcopy
from einops import rearrange, repeat

from model.BaseTransformer.spec_embed import SpecEmbedding


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def clones(module, n):
    # produce N identical layers.
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


# classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        img_path = os.path.join("img", "inp_normal", "encoding")
        os.makedirs(img_path, exist_ok=True)
        pe = self.pe[:, : x.size(1)].to("cpu").detach().numpy().copy()
        fig = plt.figure()
        plt.imshow(pe[0])
        plt.colorbar()
        plt.savefig(f"img/inp_normal/encoding/time_encoding_input_norm_lookback{x.size(1)}.png")
        """
        return self.pe[:, : x.size(1)]


class BaseTransformer(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.in_len = args.in_len
        self.out_len = args.out_len

        self.input_embedding = nn.Linear(self.num_all_features, args.d_model)
        self.spec_embedding = SpecEmbedding(args.d_model, self.num_control_features)
        self.positional_encoding = PositionalEncoding(args.d_model)  # 絶対位置エンコーディング

        self.dropout = nn.Dropout(args.dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            activation="relu",
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(args.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=args.e_layers, norm=encoder_norm)

        self.generators = nn.ModuleList([])
        for _ in range(self.out_len):
            self.generators.append(nn.Sequential(nn.LayerNorm(args.d_model), nn.Linear(args.d_model, self.num_pred_features)))

    def forward(self, input, spec):
        x = self.input_embedding(input)
        spec = self.spec_embedding(spec)
        x = torch.cat((x, spec), dim=1)
        x += self.positional_encoding(x)

        x = self.dropout(x)
        x = self.encoder(x)

        x = x.mean(dim=1)

        x_final = torch.zeros(input.shape[0], self.out_len, self.num_pred_features).to(input.device)
        for i, generator in enumerate(self.generators):
            x_final[:, i] = generator(x)

        return x_final, None
