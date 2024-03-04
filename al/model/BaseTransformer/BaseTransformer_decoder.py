import torch
from torch import nn
import math
from model.BaseTransformer.spec_embed import SpecEmbedding


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

        self.y_embedding = nn.Linear(self.num_all_features, args.d_model)

        decoder_layers = nn.TransformerDecoderLayer(
            d_model=args.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            activation="relu",
            batch_first=True,
        )
        decoder_norm = nn.LayerNorm(args.d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=args.e_layers, norm=decoder_norm)
        self.predictor = nn.Sequential(nn.Dropout(args.dropout), nn.LayerNorm(args.d_model), nn.Linear(args.d_model, self.num_pred_features))

    def forward(self, inp, spec, tgt):
        x = self.input_embedding(inp)
        x_spec = self.spec_embedding(spec)
        x = torch.cat((x, x_spec), dim=1)
        x += self.positional_encoding(x)

        tgt = torch.cat((spec, tgt), dim=2)
        y = torch.cat((inp[:, -1:, :], tgt), dim=1)[:, :-1, :]

        y = self.y_embedding(y)
        y += self.positional_encoding(y)

        mask_src, mask_tgt = self.create_mask(x, y)

        x = self.encoder(x, mask_src)

        outs = self.decoder(y, x, mask_tgt)
        outs = self.predictor(outs)

        return outs, None

    def create_mask(self, src, tgt):
        seq_len_src = src.shape[1]
        seq_len_tgt = tgt.shape[1]

        mask_tgt = self.generate_square_subsequent_mask(seq_len_tgt).to(tgt.device)
        mask_src = self.generate_square_subsequent_mask(seq_len_src).to(src.device)

        return mask_src, mask_tgt

    def generate_square_subsequent_mask(self, seq_len):
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        return mask

    def encode(self, src, spec, mask_src):
        src = self.input_embedding(src)
        spec = self.spec_embedding(spec)
        src = torch.cat((src, spec), dim=1)
        src += self.positional_encoding(src)
        src = self.encoder(src, mask_src)
        return src

    def decode(self, tgt, memory, mask_tgt):
        tgt = self.y_embedding(tgt)
        tgt += self.positional_encoding(tgt)
        tgt = self.decoder(tgt, memory, mask_tgt)
        return tgt

    def predict_func(self, inp, spec):
        seq_len_src = self.in_len + self.out_len
        mask_src = (torch.zeros(seq_len_src, seq_len_src)).type(torch.bool)
        mask_src = mask_src.float().to(inp.device)

        memory = self.encode(inp, spec, mask_src)
        outputs = inp[:, -1:, :]
        seq_len_tgt = self.out_len

        for i in range(seq_len_tgt):
            mask_tgt = (self.generate_square_subsequent_mask(outputs.size(1))).to(inp.device)

            output = self.decode(outputs, memory, mask_tgt)
            output = self.predictor(output)

            output = torch.cat([spec[:, : i + 1], output], dim=2)
            outputs = torch.cat([outputs, output[:, -1:, :]], dim=1)

        return outputs[:, -self.out_len :, -self.num_pred_features :], None
