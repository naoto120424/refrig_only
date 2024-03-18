import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ade_list = np.load("np_save_al_kc.npy")
rate_list = np.arange(0.10, 1.01, 0.10)
model_list = ["lstm", "bt", "dot", "dol"]
n_list = [21, 42, 63, 84, 105, 126, 147, 168, 189, 211]


df_LSTM = pd.DataFrame({f"{n_list[i]}": np.array(ade_list[i]) for i in range(len(rate_list))})
df_LSTM_melt = pd.melt(df_LSTM)
df_LSTM_melt["species"] = "LSTM"

df_Transformer = pd.DataFrame({f"{n_list[i]}": np.array(ade_list[i + len(rate_list)]) for i in range(len(rate_list))})
df_Transformer_melt = pd.melt(df_Transformer)
df_Transformer_melt["species"] = "Transformer"

df_dot = pd.DataFrame({f"{n_list[i]}": np.array(ade_list[i + len(rate_list) * 2]) for i in range(len(rate_list))})
df_dot_melt = pd.melt(df_dot)
df_dot_melt["species"] = "DeepOTransformer"

df_dol = pd.DataFrame({f"{n_list[i]}": np.array(ade_list[i + len(rate_list) * 3]) for i in range(len(rate_list))})
df_dol_melt = pd.melt(df_dol)
df_dol_melt["species"] = "DeepOLSTM"

df = pd.concat([df_LSTM_melt, df_Transformer_melt, df_dot_melt, df_dol_melt], axis=0)
# df = pd.concat([df_LSTM_melt, df_Transformer_melt], axis=0)
print(df.head())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x="variable", y="value", data=df, hue="species", palette="Dark2", ax=ax)

ax.set_xlabel("Number of training sequences used")
ax.set_ylabel(f"ACDS_kW[kW] ADE for 59 test data")
ax.legend(loc="best")

plt.savefig("al_KCenter.png")
