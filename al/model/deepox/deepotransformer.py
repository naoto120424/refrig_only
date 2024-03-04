import torch
from torch import nn
import math

from copy import deepcopy
from einops import rearrange


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
        return self.pe[:, : x.size(1)]


class DeepOTransformer(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.in_len = args.in_len
        self.out_len = args.out_len

        self.branch_layers = args.branch_layers
        self.trunk_layers = args.trunk_layers
        self.width = args.width

        self.input_embedding = nn.Linear(self.num_all_features, args.d_model)
        self.spec_embedding = nn.Linear(self.num_control_features, args.d_model)
        self.positional_encoding = PositionalEncoding(args.d_model)

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

        self.branch = nn.ModuleDict()
        self.branch["LinM1"] = nn.Linear(args.d_model, self.out_len * self.num_pred_features * self.width)
        self.branch["NonM1"] = nn.ReLU()
        for i in range(2, self.branch_layers):
            self.branch["LinM{}".format(i)] = nn.Linear(self.out_len * self.num_pred_features * self.width, self.out_len * self.num_pred_features * self.width)
            self.branch["NonM{}".format(i)] = nn.ReLU()
        self.branch["LinMout"] = nn.Linear(self.out_len * self.num_pred_features * self.width, self.out_len * self.num_pred_features * self.width)

        self.trunk = nn.ModuleDict()
        self.trunk["LinM1"] = nn.Linear(self.num_all_features, self.num_pred_features * self.width)
        self.trunk["NonM1"] = nn.ReLU()
        for i in range(2, self.trunk_layers + 1):
            self.trunk["LinM{}".format(i)] = nn.Linear(self.num_pred_features * self.width, self.num_pred_features * self.width)
            self.trunk["NonM{}".format(i)] = nn.ReLU()

        self.params = nn.ParameterDict()
        self.params["bias"] = nn.Parameter(torch.zeros([self.num_pred_features]))

    def forward(self, inp, spec):
        x = self.input_embedding(inp)
        x_spec = self.spec_embedding(spec)
        x = torch.cat((x, x_spec), dim=1)
        x += self.positional_encoding(x)

        x = self.encoder(x)
        x = x.mean(dim=1)

        if self.branch_layers == 1:
            LinM = self.branch["LinM1"]
            x = LinM(x)
            x = rearrange(x, "bs (out_len n d) -> bs out_len n d", out_len=self.out_len, n=self.num_pred_features)
        else:
            for i in range(1, self.branch_layers):
                LinM = self.branch["LinM{}".format(i)]
                NonM = self.branch["NonM{}".format(i)]
                x = NonM(LinM(x))
            x = self.branch["LinMout"](x)
            x = rearrange(x, "bs (out_len n d) -> bs out_len n d", out_len=self.out_len, n=self.num_pred_features)

        y = inp[:, -1:, :]

        for t in range(1, self.out_len + 1):
            y_out = y
            for i in range(1, self.trunk_layers + 1):
                LinM = self.trunk["LinM{}".format(i)]
                NonM = self.trunk["NonM{}".format(i)]
                y_out = NonM(LinM(y_out))

            y_out = rearrange(y_out, "bs l (n d) -> bs l n d", n=self.num_pred_features)
            y_out = torch.sum(x[:, :t] * y_out, dim=-1, keepdim=False) + self.params["bias"]

            y_out = torch.cat((spec[:, :t], y_out), dim=-1)
            y = torch.cat((y, y_out[:, -1:]), dim=1)

        return y[:, 1:, self.num_control_features :]
