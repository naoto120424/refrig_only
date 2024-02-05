import torch
from torch import nn
from einops import rearrange, repeat


class DeepOLSTM(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.in_len = args.in_len
        self.out_len = args.out_len

        self.branch = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model, args.d_model))

        self.trunk = nn.ModuleList([])
        self.trunk.append(
            nn.ModuleList(
                [
                    nn.Linear(1, args.trunk_d_model),
                    nn.ReLU()
                ]
            )
        )
        for _ in range(args.trunk_layers-2):
            self.trunk.append(
                nn.ModuleList(
                    [
                        nn.Linear(args.trunk_d_model, args.trunk_d_model),
                        nn.ReLU(),
                    ]
                )
            )
        self.trunk.append(
            nn.ModuleList(
                [
                    nn.Linear(args.trunk_d_model, args.d_model),
                    nn.ReLU(),
                ]
            )
        )

        self.lstm = nn.LSTM(
            input_size=cfg.NUM_ALL_FEATURES,
            hidden_size=args.d_model,
            num_layers=args.e_layers,
            dropout=args.dropout,
            batch_first=True,
        )
        self.spec_dense = nn.Linear(cfg.NUM_CONTROL_FEATURES, args.d_model)

    def forward(self, inp, spec, timedata, h=None):
        hidden1, _ = self.lstm(inp, h)
        hidden2 = self.spec_dense(spec)

        x = torch.cat([hidden1, hidden2], dim=1)
        x = x.mean(dim=1)
        x = repeat(x, "bs d -> bs out_len n_pred d", out_len=self.out_len, n_pred=self.num_pred_features)
        x = self.branch(x)

        y = repeat(timedata, "bs out_len -> bs out_len n_pred 1", out_len=self.out_len, n_pred=self.num_pred_features)

        for linear, relu in self.trunk:
            y = linear(y)
            y = relu(y)

        # print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        x = torch.sum(x * y, dim=-1, keepdim=False)

        return x, None
