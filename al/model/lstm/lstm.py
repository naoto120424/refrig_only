import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, cfg, args):
        super(LSTMClassifier, self).__init__()
        self.input_dim = cfg.NUM_ALL_FEATURES
        self.spec_dim = cfg.NUM_CONTROL_FEATURES
        self.output_dim = cfg.NUM_PRED_FEATURES
        self.out_len = args.out_len
        self.num_hidden_units = args.d_model
        self.num_layers = args.e_layers
        self.dropout = args.dropout

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.num_hidden_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.spec_dense = nn.Linear(self.spec_dim, self.num_hidden_units)
        self.linear = nn.Sequential(
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            nn.ReLU(inplace=True),
        )

        self.generators = nn.ModuleList([])
        for _ in range(self.out_len):
            self.generators.append(
                nn.Sequential(nn.LayerNorm(self.num_hidden_units), nn.Linear(self.num_hidden_units, self.output_dim))
            )

    def forward(self, x, spec, h=None):
        hidden1, _ = self.lstm(x, h)
        hidden2 = self.spec_dense(spec)
        y = self.linear(torch.cat([hidden1, hidden2], dim=1))
        y = y.mean(dim=1)

        y_final = torch.zeros(x.shape[0], self.out_len, self.output_dim).to(x.device)
        for i, generator in enumerate(self.generators):
            y_final[:, i] = generator(y)

        return y_final, y
