import subprocess
import numpy as np

model_list = [
    "LSTM",
    "BaseTransformer",
    # "BaseTransformer_sensor_first",
    # "BaseTransformer_3types_aete",
    "BaseTransformer_3types_AgentAwareAttention",
    # "BaseTransformer_5types_AgentAwareAttention",
    # "BaseTransformer_individually_aete",
    # "BaseTransformer_individually_AgentAwareAttention",
    # "Transformer",
    # "Linear",
    # "NLinear",
    # "DLinear",
    # "DeepONet",
    "DeepOTransformer",
    "DeepOLSTM",
    # "s4",
    # "s4d"
    "Crossformer",
]

bs = 128
epoch = 100
criterion = "MSE"
patience = 100
delta = 0.0
lr = 1e-3

""" for all model """
in_len_list = [10]
out_len_list = [1]
e_layers_list = [3]
d_model_list = [512]
dropout_list = [0.1]

""" for transformer & crossformer & deepotransformer"""
n_heads_list = [8]

""" for crossformer """
seg_len_list = [2]
win_size_list = [2]
factor_list = [3]

""" for deepotransformer """
trunk_layers_list = [2]
# trunk_dim_list = [512, 1024]

""" params list create """
params_list = []
for model in model_list:
    for in_len in in_len_list:
        for out_len in out_len_list:
            for e_layers in e_layers_list:
                for d_model in d_model_list:
                    for dropout in dropout_list:
                        if "LSTM" in model:
                            if "DeepO" in model:
                                for trunk_layers in trunk_layers_list:
                                    params_list.append([model, in_len, out_len, e_layers, d_model, dropout, 0, d_model * 2, 0, 0, 0, trunk_layers, d_model])
                            else:
                                params_list.append([model, in_len, out_len, e_layers, d_model, dropout, 0, 0, 0, 0, 0, 0, 0])
                            
                        elif "former" in model:
                            for n_head in n_heads_list:
                                if "BaseTransformer" in model:
                                    params_list.append([model, in_len, out_len, e_layers, d_model, dropout, n_head, d_model * 2, 0, 0, 0, 0, 0])
                                elif "Crossformer" in model:
                                    for seg_len in seg_len_list:
                                        for win_size in win_size_list:
                                            for factor in factor_list:
                                                params_list.append([model, in_len, out_len, e_layers, d_model, dropout, n_head, d_model * 2, seg_len, win_size, factor, 0, 0])
                                elif "DeepOTransformer" in model:
                                    for trunk_layers in trunk_layers_list:
                                        params_list.append([model, in_len, out_len, e_layers, d_model, dropout, n_head, d_model * 2, 0, 0, 0, trunk_layers, d_model])
                        
                        elif "Linear" in model:
                            params_list.append([model, in_len, out_len, e_layers, d_model, dropout, 0, 0, 0, 0, 0, 0, 0])
                        
                        elif "s4" in model:
                            params_list.append([model, in_len, out_len, e_layers, d_model, dropout, 0, 0, 0, 0, 0, 0, 0])
                            
for i, params in enumerate(params_list):
    command = [
        "python",
        "train.py",
        "--e_name",
        f"20240124 step3 No Earlystopping",
        "--bs",
        str(bs),
        "--train_epochs",
        str(epoch),
        "--model",
        str(params[0]),
        "--in_len",
        str(params[1]),
        "--out_len",
        str(params[2]),
        "--e_layers",
        str(params[3]),
        "--d_model",
        str(params[4]),
        "--dropout",
        str(params[5]),
        "--n_heads",
        str(params[6]),
        "--d_ff",
        str(params[7]),
        "--seg_len",
        str(params[8]),
        "--win_size",
        str(params[9]),
        "--factor",
        str(params[10]),
        "--trunk_layers",
        str(params[11]),
        "--trunk_d_model",
        str(params[12]),
        "--patience",
        str(patience),
        "--delta",
        str(delta),
        # "--debug",
        # str(True),
    ]

    print(f"command {i+1}: {command}\n")

    subprocess.run(command)
    print("\n\n")