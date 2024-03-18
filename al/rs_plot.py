import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def vis(df, n_method, n_m):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(x="variable", y="value", data=df, hue="species", palette="Dark2", ax=ax)

    ax.set_title(f"{n_method} {str(df['species'][1])}")
    ax.set_xlabel("Number of training sequences used")
    ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
    ax.legend(loc="best")

    plt.savefig(f"al_{n_m}_{str(df['species'][1])}.png", dpi=200)
    plt.close()


rate_list = np.arange(0.10, 1.01, 0.10)
model_list = ["lstm", "bt", "dot", "dol"]
n_list = [21, 42, 63, 84, 105, 126, 147, 168, 189, 211]

""" random sampling """
ade_list_rs = np.load("np_save_rs.npy")

df_LSTM_rs = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_rs[i * len(model_list)]) for i in range(len(rate_list))})
df_LSTM_melt_rs = pd.melt(df_LSTM_rs)
df_LSTM_melt_rs["species"] = "LSTM"

df_Transformer_rs = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_rs[i * len(model_list) + 1]) for i in range(len(rate_list))})
df_Transformer_melt_rs = pd.melt(df_Transformer_rs)
df_Transformer_melt_rs["species"] = "Transformer"

df_dot_rs = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_rs[i * len(model_list) + 2]) for i in range(len(rate_list))})
df_dot_melt_rs = pd.melt(df_dot_rs)
df_dot_melt_rs["species"] = "DeepOTransformer"

df_dol_rs = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_rs[i * len(model_list) + 3]) for i in range(len(rate_list))})
df_dol_melt_rs = pd.melt(df_dol_rs)
df_dol_melt_rs["species"] = "DeepOLSTM"

df = pd.concat([df_LSTM_melt_rs, df_Transformer_melt_rs, df_dot_melt_rs, df_dol_melt_rs], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="species", palette="Dark2", ax=ax)

ax.set_title("Random Sampling")
ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("al_RandomSampling.png", dpi=200)
plt.close()

n_method, n_m = "Random Sampling", "rs"
vis(df_LSTM_melt_rs, n_method, n_m)
vis(df_Transformer_melt_rs, n_method, n_m)
vis(df_dot_melt_rs, n_method, n_m)
vis(df_dol_melt_rs, n_method, n_m)

""" KCenter """
ade_list_kc = np.load("np_save_al_kc.npy")
df_LSTM_kc = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_kc[i]) for i in range(len(rate_list))})
df_LSTM_melt_kc = pd.melt(df_LSTM_kc)
df_LSTM_melt_kc["species"] = "LSTM"

df_Transformer_kc = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_kc[i + len(rate_list)]) for i in range(len(rate_list))})
df_Transformer_melt_kc = pd.melt(df_Transformer_kc)
df_Transformer_melt_kc["species"] = "Transformer"

df_dot_kc = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_kc[i + len(rate_list) * 2]) for i in range(len(rate_list))})
df_dot_melt_kc = pd.melt(df_dot_kc)
df_dot_melt_kc["species"] = "DeepOTransformer"

df_dol_kc = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_kc[i + len(model_list) * 3]) for i in range(len(rate_list))})
df_dol_melt_kc = pd.melt(df_dol_kc)
df_dol_melt_kc["species"] = "DeepOLSTM"

df = pd.concat([df_LSTM_melt_kc, df_Transformer_melt_kc, df_dot_melt_kc, df_dol_melt_kc], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="species", palette="Dark2", ax=ax)

ax.set_title("KCenter")
ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("al_KCenter.png", dpi=200)
plt.close()

n_method, n_m = "KCenter", "kc"
vis(df_LSTM_melt_kc, n_method, n_m)
vis(df_Transformer_melt_kc, n_method, n_m)
vis(df_dot_melt_kc, n_method, n_m)
vis(df_dol_melt_kc, n_method, n_m)

""" Learning Loss """
ade_list_ll = np.load("np_save.npy")

df_LSTM_ll = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_ll[i]) for i in range(len(rate_list))})
df_LSTM_melt_ll = pd.melt(df_LSTM_ll)
df_LSTM_melt_ll["species"] = "LSTM"

df_Transformer_ll = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_ll[i + len(rate_list)]) for i in range(len(rate_list))})
df_Transformer_melt_ll = pd.melt(df_Transformer_ll)
df_Transformer_melt_ll["species"] = "Transformer"

df_dot_ll = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_ll[i + len(rate_list) * 2]) for i in range(len(rate_list))})
df_dot_melt_ll = pd.melt(df_dot_ll)
df_dot_melt_ll["species"] = "DeepOTransformer"

df_dol_ll = pd.DataFrame({f"{n_list[i]}": np.array(ade_list_ll[i + len(rate_list) * 3]) for i in range(len(rate_list))})
df_dol_melt_ll = pd.melt(df_dol_ll)
df_dol_melt_ll["species"] = "DeepOLSTM"

df = pd.concat([df_LSTM_melt_ll, df_Transformer_melt_ll, df_dot_melt_ll, df_dol_melt_ll], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="species", palette="Dark2", ax=ax)

ax.set_title("Learning Loss")
ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("al_LearningLoss.png", dpi=200)
plt.close()

n_method, n_m = "Learning Loss", "ll"
vis(df_LSTM_melt_ll, n_method, n_m)
vis(df_Transformer_melt_ll, n_method, n_m)
vis(df_dot_melt_ll, n_method, n_m)
vis(df_dol_melt_ll, n_method, n_m)

""" Transformer """
df_Transformer_melt_rs["method"] = "Random Sampling"
df_Transformer_melt_kc["method"] = "KCenter"
df_Transformer_melt_ll["method"] = "Learning Loss"
df = pd.concat([df_Transformer_melt_rs, df_Transformer_melt_kc, df_Transformer_melt_ll], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="method", palette="Dark2", ax=ax)

ax.set_title("Transformer")
ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("summary_Transformer.png", dpi=200)
plt.close()

""" LSTM """
df_LSTM_melt_rs["method"] = "Random Sampling"
df_LSTM_melt_kc["method"] = "KCenter"
df_LSTM_melt_ll["method"] = "Learning Loss"
df = pd.concat([df_LSTM_melt_rs, df_LSTM_melt_kc, df_LSTM_melt_ll], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="method", palette="Dark2", ax=ax)

ax.set_title("LSTM")
ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("summary_LSTM.png", dpi=200)
plt.close()

""" DOT """
df_dot_melt_rs["method"] = "Random Sampling"
df_dot_melt_kc["method"] = "KCenter"
df_dot_melt_ll["method"] = "Learning Loss"
df = pd.concat([df_dot_melt_rs, df_dot_melt_kc, df_dot_melt_ll], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="method", palette="Dark2", ax=ax)

ax.set_title("DeepOTransformer")
ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("summary_DOT.png", dpi=200)
plt.close()

""" DOL """
df_dol_melt_rs["method"] = "Random Sampling"
df_dol_melt_kc["method"] = "KCenter"
df_dol_melt_ll["method"] = "Learning Loss"
df = pd.concat([df_dol_melt_rs, df_dol_melt_kc, df_dol_melt_ll], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="method", palette="Dark2", ax=ax)

ax.set_title("DeepOLSTM")
ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("summary_DOL.png", dpi=200)
plt.close()
