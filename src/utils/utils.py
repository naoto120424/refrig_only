import torch
import torch.nn as nn
import numpy as np
import random
import os

from model.lstm.lstm import LSTMClassifier
from model.BaseTransformer.base_transformer import BaseTransformer
from model.deepox.deepotransformer import DeepOTransformer
from model.deepox.deepolstm import DeepOLSTM


class CFG:

    def __init__(self, args) -> None:
        if args.dataset == "teacher":
            self.NUM_CONTROL_FEATURES = 9
            self.NUM_PRED_FEATURES = 41
            self.NUM_BYPRODUCT_FEATURES = 37
            self.NUM_TARGET_FEATURES = 4
            self.NUM_ALL_FEATURES = 50
            self.DATA_PATH = os.path.join("..", "data_step2")
        else:
            self.NUM_CONTROL_FEATURES = 15
            self.NUM_PRED_FEATURES = 47
            self.NUM_BYPRODUCT_FEATURES = 40
            self.NUM_TARGET_FEATURES = 7
            self.NUM_ALL_FEATURES = 62
            self.DATA_PATH = os.path.join("..", "data_refrig_only")

        self.RESULT_PATH = os.path.join("..", "result")
        self.MLFLOW_PATH = os.path.join("..", "mlflow")


model_list = {
    "lstm",
    "bt",
    "cf",
    "dot",
    "dol",
    "s4",
    "s4d",
}

criterion_list = {"mse": nn.MSELoss(), "l1": nn.L1Loss()}

predict_time_list = []


def deviceDecision():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def seed_everything(seed: int = 42) -> None:
    """
    ランダムシードを設定し、再現性を担保するために必要な関数。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# model decide from model name
def modelDecision(args, cfg):
    if args.model == "lstm":
        return LSTMClassifier(cfg, args)
    elif args.model == "bt":
        return BaseTransformer(cfg, args)
    elif args.model == "dot":
        return DeepOTransformer(cfg, args)
    elif args.model == "dol":
        return DeepOLSTM(cfg, args)

    raise ValueError("unknown model name")
