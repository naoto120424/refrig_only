import torch
from torch import nn
import math
from model.BaseTransformer.spec_embed import SpecEmbedding

# classes
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
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
        self.positional_embedding = PositionalEmbedding(args.d_model)  # 絶対位置エンコーディング
        
        self.dropout = nn.Dropout(args.dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff, 
            dropout=args.dropout,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=args.e_layers
        )
        
        self.y_embedding = nn.Linear(self.num_pred_features, args.d_model)
        
        decoder_layers = nn.TransformerDecoderLayer(
            d_model = args.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layers,
            num_layers=args.e_layers
        )
        self.predictor = nn.Sequential(nn.LayerNorm(args.d_model), nn.Linear(args.d_model, self.num_pred_features))
        
    def forward(self, inp, spec):
        x = self.input_embedding(inp)
        spec = self.spec_embedding(spec)
        x = torch.cat((x, spec), dim=1)
        
        x += self.positional_embedding(x)

        x = self.dropout(x)
        x = self.encoder(x)
        
        y = inp[:, -1, -self.num_pred_features:].unsqueeze(1)
        
        for _ in range(self.out_len):
            y_embed = self.y_embedding(y)
            x_dec = self.decoder(x, y_embed)
            x_dec = x_dec.mean(dim=1)
            x_dec = self.predictor(x_dec).unsqueeze(1)
            y = torch.cat((y, x_dec), dim=1)
        
        return y[:, 1:], None