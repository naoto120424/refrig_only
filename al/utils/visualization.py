import numpy as np
import matplotlib.pyplot as plt
import os, mlflow

from sklearn.metrics import mean_absolute_error

class Eva():
    def __init__(self, data):
        self.pred_name = data["pred_name"]
        self.pred_unit = data["pred_unit"]
        self.target_name = data["target_name"]
        self.target_unit = data["target_unit"]

        self.evaluation_list = ["ade", "fde"]
        self.eva_all = []

        """
            ade: average displacement error (Average of all time errors)
            fde: final displacement error (Error of the last time)
        """

    # Calculate Evaluation
    def evaluation(self, in_len, gt_array, pred_array):
        eva_case = []
        for i in range(len(self.pred_name)):
            ade = mean_absolute_error(np.array(gt_array)[in_len:, i], np.array(pred_array)[in_len:, i])
            fde = abs(gt_array[-1][i] - pred_array[-1][i])

            eva_case.append(ade)
            eva_case.append(fde)

        self.eva_all.append(eva_case)

    # Save Evaluation
    def save_evaluation(self):
        for i, target in enumerate(self.pred_name):
            for j, evaluation in enumerate(self.evaluation_list):
                np_array = np.array(self.eva_all)[:, j + len(self.evaluation_list) * i]
                mlflow.log_metric(f"{target}_{evaluation.upper()}", np.mean(np_array))

    # Make Graph
    def visualization(self, gt_array, pred_array, case_name, result_path):
        case_path = os.path.join(result_path, "vis", case_name)
        img_path = os.path.join(case_path, "img")
        img_no_yticks = os.path.join(case_path, "img_no_yticks")
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(img_no_yticks, exist_ok=True)

        for i in range(len(self.pred_name)):
            for j in range(2):
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(np.array(gt_array)[:, i], color="#e46409", label="gt")
                ax.plot(np.array(pred_array)[:, i], color="b", label="pred")
                ax.set_title(case_name, fontsize=16)
                ax.set_xlabel("Time[s]", fontsize=16)
                ax.set_ylabel(f"{self.pred_name[i]} [{self.pred_unit[i]}]", fontsize=16)
                ax.legend(loc="best", fontsize=16)
                if j == 0:
                    ax.set_yticks([])
                    plt.tight_layout()
                    plt.savefig(os.path.join(img_no_yticks, f"{self.pred_name[i]}.png"))
                else:
                    plt.tight_layout()
                    plt.savefig(os.path.join(img_path, f"{self.pred_name[i]}.png"))
                plt.close()

        # For PowerPoint Slide
        grf_row = 3
        grf_col = 3
        fig = plt.figure(figsize=(12, 8))
        ax_list = []
        for j in range(grf_row):
            for i in range(grf_col):
                if (j * grf_col + i) < len(self.target_name):
                    ax_list.append(fig.add_subplot(grf_row, grf_col, j * grf_col + i + 1))
                    ax_list[j * grf_col + i].plot(np.array(gt_array)[:, self.pred_name.index(self.target_name[j*grf_col+i])], color="#e46409", label="gt")
                    ax_list[j * grf_col + i].plot(np.array(pred_array)[:, self.pred_name.index(self.target_name[j*grf_col+i])], color="b", label="pred")
                    ax_list[j * grf_col + i].set_xlabel(f"Time[s]"), 
                    ax_list[j * grf_col + i].set_ylabel(f"{self.target_name[j * grf_col + i]} [{self.target_unit[j * grf_col + i]}]")
                    ax_list[j * grf_col + i].set_yticks([])
                    ax_list[j * grf_col + i].legend(loc="best")

        fig.tight_layout()
        plt.savefig(os.path.join(case_path, f"target_kW.png"))
        plt.close()


def print_model_summary(args, device, len_train_index, len_val_index):
    print("\n\nTrain Start")
    print("----------------------------------------------")
    print(f"Dataset     : {str.upper(args.dataset)}")
    print(f"Device      : {str.upper(device)}")
    print(f"Train Case  : {len_train_index}")
    print(f"Val Case    : {len_val_index}")
    print(f"Criterion   : {str.upper(args.criterion)}")
    print(f"Batch Size  : {args.bs}")
    print(f"in_len      : {args.in_len}")
    print(f"out_len     : {args.out_len}")
    print(f"Model       : {args.model}")
    print(f" - e_layers  : {args.e_layers}")
    print(f" - d_model   : {args.d_model}")
    print(f" - dropout   : {args.dropout}")
    if args.model == "bt":
        print(f" - num heads : {args.n_heads}")
        print(f" - dim of ff : {args.d_ff}")

    if "do" in args.model:
        if "t" in args.model:
            print(f" - num heads : {args.n_heads}")
            print(f" - dim of ff : {args.d_ff}")
        print(f" - branch layers : {args.branch_layers}")
        print(f" - trunk layers  : {args.trunk_layers}")
        print(f" - width         : {args.width}")


def mlflow_summary(cfg, args):
    mlflow.set_tracking_uri(cfg.MLFLOW_PATH)
    mlflow.set_experiment(args.e_name)
    mlflow.start_run()
    mlflow.log_param("dataset", args.dataset)
    mlflow.log_param("model", args.model)
    mlflow.log_param("debug", args.debug)
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("batch size", args.bs)
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("patience", args.patience)
    mlflow.log_param("delta", args.delta)
    mlflow.log_param("in_len", args.in_len)
    mlflow.log_param("out_len", args.out_len)
    mlflow.log_param("d_model", args.d_model)
    mlflow.log_param("e_layers", args.e_layers)
    mlflow.log_param("dropout", args.dropout)
    if args.model == "bt":
        mlflow.log_param("heads", args.n_heads)
        mlflow.log_param("d_ff", args.d_ff)
    if "do" in args.model:
        if "t" in args.model:
            mlflow.log_param("heads", args.n_heads)
            mlflow.log_param("d_ff", args.d_ff)
        mlflow.log_param("branch layers", args.branch_layers)
        mlflow.log_param("trunk layers", args.trunk_layers)
        mlflow.log_param("width", args.width)
