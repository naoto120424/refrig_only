import torch
import torch.nn as nn
import numpy as np
import random
import os


class CFG:
    NUM_CONTROL_FEATURES = 15
    NUM_PRED_FEATURES = 47
    NUM_BYPRODUCT_FEATURES = 40
    NUM_TARGET_FEATURES = 7
    NUM_ALL_FEATURES = 62
    DATA_PATH = os.path.join("..", "refrig_only_data")
    RESULT_PATH = os.path.join("..", "result")
    MLFLOW_PATH = os.path.join("..", "mlflow_experiment")


model_list = {
    "LSTM",
    "BaseTransformer",
    "BaseTransformer_pred_token",
    "BaseTransformer_sensor_first",
    "BaseTransformer_3types_aete",
    "BaseTransformer_3types_AgentAwareAttention",
    "BaseTransformer_5types_AgentAwareAttention",
    "BaseTransformer_individually_aete",
    "BaseTransformer_individually_AgentAwareAttention",
    "Transformer",
    "Crossformer",
    "Linear",
    "DLinear",
    "NLinear",
    "DeepOTransformer",
    "DeepOLSTM",
    "s4",
    "s4d"
}

criterion_list = {"MSE": nn.MSELoss(), "L1": nn.L1Loss()}

predict_time_list = []


def deviceDecision():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# model decide from model name
def modelDecision(args, cfg):
    if "BaseTransformer" in args.model:
        if args.model == "BaseTransformer":
            from model.BaseTransformer.base_transformer import BaseTransformer
        if args.model == "BaseTransformer_pred_token":
            from model.BaseTransformer.base_transformer_pred_token import BaseTransformer
        elif args.model == "BaseTransformer_sensor_first":
            from model.BaseTransformer.base_transformer_sensor_first import BaseTransformer
        elif args.model == "BaseTransformer_3types_aete":
            from model.BaseTransformer.base_transformer_3types_aete import BaseTransformer
        elif args.model == "BaseTransformer_3types_AgentAwareAttention":
            from model.BaseTransformer.base_transformer_3types_AgentAwareAttention import BaseTransformer
        elif args.model == "BaseTransformer_5types_AgentAwareAttention":
            from model.BaseTransformer.base_transformer_5types_AAA import BaseTransformer
        elif args.model == "BaseTransformer_individually_aete":
            from model.BaseTransformer.base_transformer_individually_aete import BaseTransformer
        elif args.model == "BaseTransformer_individually_AgentAwareAttention":
            from model.BaseTransformer.base_transformer_individually_AgentAwareAttention import BaseTransformer

        return BaseTransformer(cfg, args)

    if "Deep" in args.model:
        if "DeepOTransformer" in args.model:
            from model.DeepOTransformer.deepotransformer import DeepOTransformer
            return DeepOTransformer(cfg, args)
        
        elif args.model == "DeepOLSTM":
            from model.DeepOTransformer.deepolstm import DeepOLSTM
            return DeepOLSTM(cfg, args)
    

    if "Transformer" in args.model:
        if args.model == "Transformer":
            from model.Transformer.transformer import Transformer
        return Transformer(cfg, args)

    if "Crossformer" in args.model:
        if args.model == "Crossformer":
            from model.crossformer.cross_former import Crossformer
        return Crossformer(cfg, args)

    if "Linear" in args.model:
        if args.model == "Linear":
            from model.linear.linear import Model
        elif args.model == "NLinear":
            from model.linear.nlinear import Model
        elif args.model == "DLinear":
            from model.linear.dlinear import Model

        return Model(cfg, args)
    
    if "LSTM" in args.model:
        from model.lstm.lstm import LSTMClassifier
        return LSTMClassifier(cfg, args)
    
    
    if "s4" in args.model:
        if args.d_model == "s4d":
            from model.s4.s4d import S4D
            return S4D(args.d_model)
        
        
        elif args.model == "s4":
            from model.s4.s4 import S4Block
            return S4Block(args.d_model)
    


    return None