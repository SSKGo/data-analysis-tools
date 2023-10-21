# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 学校保健統計調査 / 令和2年度 都道府県表
# [Download](https://www.e-stat.go.jp/stat-search/files?page=1&query=%E8%BA%AB%E9%95%B7%E3%83%BB%E4%BD%93%E9%87%8D%E3%81%AE%E5%B9%B3%E5%9D%87%E5%80%A4&layout=dataset&stat_infid=000032108465)

# %%
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from sklearn.preprocessing import LabelEncoder

excel_file_path = "./data/r2_hoken_tokei_05.xls"

# Load sheet name list and age by xlrd as preprocess
wb = xlrd.open_workbook(excel_file_path)
sheet_names = wb.sheet_names()
# 1 sheet is for 1 age.
# age is stored in cell(3, 1) in each sheet.
ages = []
for sheet_name in sheet_names:
    sheet = wb.sheet_by_name(sheet_name)
    ages.append(sheet.cell_value(3, 1))
wb.release_resources()
del wb
# Remove "歳" which is the last character.
ages = [int(age_str[:-1]) for age_str in ages]

# Merge all sheet data in 1 data frame.
list_df = []
for age, sheet_name in zip(ages, sheet_names):
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=6)
    for col_slice, sex in zip([[1, 2, 4], [1, 6, 8]], ["Male", "Female"]):
        df_sex = df.iloc[:-2, col_slice]
        df_sex.columns = ["Prefecture", "Height", "Weight"]
        df_sex["Sex"] = sex
        df_sex["Age"] = age
        list_df.append(df_sex)
df_all = pd.concat(list_df)
df_all = df_all[df_all["Prefecture"] != "全　　　国"]


# %%
# Visualization
fig = plt.figure(tight_layout=True, figsize=(20, 16))
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
axes4 = []
axes4.append(plt.subplot2grid((3, 2), (2, 0)))
axes4.append(plt.subplot2grid((3, 2), (2, 1)))
df_prefec = df_all.pivot(index=("Sex", "Age"), columns="Prefecture", values="Weight")

for i_num, (sex, color) in enumerate(zip(["Male", "Female"], ["blue", "red"])):
    df_sex = df_all[df_all["Sex"] == sex]
    df_sex.plot.scatter(x="Age", y="Weight", alpha=0.5, ax=ax1, color=color, label=sex)
    df_sex.plot.scatter(
        x="Height", y="Weight", alpha=0.5, ax=ax2, color=color, label=sex
    )
    boxprops = dict(linestyle="--", color=color)
    medianprops = dict(linestyle="-", color=color)
    ax3.boxplot(df_prefec.loc[sex, :], boxprops=boxprops, medianprops=medianprops)
    ax3.set_xlabel("Prefecture")
    ax3.set_ylabel("Weight")
    groups = df_sex.groupby(df_sex["Prefecture"])
    for _, group in groups:
        axes4[i_num].plot(group["Height"], group["Weight"])
        axes4[i_num].set_xlabel("Height")
        axes4[i_num].set_ylabel("Weight")
    axes4[i_num].set_title(sex)
ax1.legend()
ax2.legend()
fig.show()


# %%
# Encode categorical variables
df_all_encode = df_all.copy()

label_encoder_sex = LabelEncoder()
df_all_encode["Sex"] = label_encoder_sex.fit_transform(df_all["Sex"]) + 1

label_encoder_prefecture = LabelEncoder()
df_all_encode["Prefecture"] = (
    label_encoder_prefecture.fit_transform(df_all["Prefecture"]) + 1
)

# print(label_encoder_sex.classes_)
# print(label_encoder_prefecture.classes_)

display(df_all_encode.head())


# %%
import pystan

script = """
data {
    int N;
    int N_p;
    real X[N];
    real Y[N];
    int<lower=0, upper=N_p> p_id[N];
}

parameters {
    real a0;
    real b0;
    real a_p[N_p];
    real b_p[N_p];
    real<lower=0> sgm_a;
    real<lower=0> sgm_b;
    real<lower=0> sgm_Y;
}

transformed parameters {
    real a[N_p];
    real b[N_p];
    for (i in 1:N_p){
        a[i] = a0 + a_p[i];
        b[i] = b0 + b_p[i];
    }
}

model {
    for (i in 1:N_p){
        a_p[i] ~ normal(0, sgm_a);
        b_p[i] ~ normal(0, sgm_b);
    }

    for (i in 1:N){
        Y[i] ~ normal(a[p_id[i]] * X[i] + b[p_id[i]], sgm_Y);
    }
}
"""

sm = pystan.StanModel(model_code=script)

stan_data = {
    "N": df_all_encode.shape[0],
    "N_p": len(df_all_encode["Prefecture"].unique()),
    "X": df_all_encode["Age"].tolist(),
    "Y": df_all_encode["Weight"].tolist(),
    "p_id": df_all_encode["Prefecture"].tolist(),
}


# %%
import pystan

script = """
data {
    int N;
    int N_sex;
    real X[N];
    real Y[N];
    int<lower=0, upper=N_sex> i_sex[N];
}

parameters {
    real a0;
    real b0;
    real c0;
    real a_sex[N_sex];
    real b_sex[N_sex];
    real c_sex[N_sex];
    real<lower=0> sgm_a;
    real<lower=0> sgm_b;
    real<lower=0> sgm_c;
    real<lower=0> sgm_Y;
}

transformed parameters {
    real a[N_sex];
    real b[N_sex];
    real c[N_sex];
    for (i in 1:N_sex){
        a[i] = a0 + a_sex[i];
        b[i] = b0 + b_sex[i];
        c[i] = c0 + c_sex[i];
    }
}

model {
    for (i in 1:N_sex){
        a_sex[i] ~ normal(0, sgm_a);
        b_sex[i] ~ normal(0, sgm_b);
        c_sex[i] ~ normal(0, sgm_c);
    }

    for (i in 1:N){
        Y[i] ~ normal(a[i_sex[i]] * square(X[i]) + b[i_sex[i]] * X[i] + c[i_sex[i]], sgm_Y);
    }
}
"""

sm = pystan.StanModel(model_code=script)

stan_data = {
    "N": df_all_encode.shape[0],
    "N_sex": len(df_all_encode["Sex"].unique()),
    "X": df_all_encode["Height"].tolist(),
    "Y": df_all_encode["Weight"].tolist(),
    "i_sex": df_all_encode["Sex"].tolist(),
}


# %%
import pystan

script = """
data {
    int N;
    int N_sex;
    real X[N];
    real Y[N];
    int<lower=0, upper=N_sex> i_sex[N];
}

parameters {
    real a_sex[N_sex];
    real b_sex[N_sex];
    real c_sex[N_sex];
    real<lower=0> sgm_a;
    real<lower=0> sgm_b;
    real<lower=0> sgm_c;
    real<lower=0> sgm_Y;
}

model {
    for (i in 1:N_sex){
        a_sex[i] ~ normal(0, sgm_a);
        b_sex[i] ~ normal(0, sgm_b);
        c_sex[i] ~ normal(0, sgm_c);
    }

    for (i in 1:N){
        Y[i] ~ normal(a_sex[i_sex[i]] * square(X[i]) + b_sex[i_sex[i]] * X[i] + c_sex[i_sex[i]], sgm_Y);
    }
}
"""

sm = pystan.StanModel(model_code=script)

stan_data = {
    "N": df_all_encode.shape[0],
    "N_sex": len(df_all_encode["Sex"].unique()),
    "X": df_all_encode["Age"].tolist(),
    "Y": df_all_encode["Weight"].tolist(),
    "i_sex": df_all_encode["Sex"].tolist(),
}


# %%
fit = sm.sampling(
    data=stan_data,
    n_jobs=1,
    iter=8000,
    warmup=2000,
    chains=3,
    seed=1,
    control=dict(adapt_delta=0.9, max_treedepth=15),
)


# %%
print(fit)


# %%
import arviz

arviz.plot_trace(fit)


# %%
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-darkgrid")
x_graph = np.linspace(4, 18, 15)

i_sex = 2

df = fit.to_dataframe()
y_chains = []
for i_chain in df["chain"].unique():
    df_chain = df[df["chain"] == i_chain]
    # y_est = df_chain["b0"].values + df_chain["b1"].values * x_graph[0]
    # y_est = df_chain["b0"].values.reshape(-1, 1) + df_chain["b1"].values.reshape(-1, 1) * x_graph.reshape(-1, 1).T
    y_est = (
        df_chain[f"c_sex[{i_sex}]"].values.reshape(-1, 1)
        + df_chain[f"b_sex[{i_sex}]"].values.reshape(-1, 1) * x_graph.reshape(-1, 1).T
        + df_chain[f"a_sex[{i_sex}]"].values.reshape(-1, 1)
        * (x_graph.reshape(-1, 1) * x_graph.reshape(-1, 1)).T
    )
    y_chains.append(y_est)
y_graph = np.stack(y_chains, axis=0)

# Merge chain again to calculate mean
y_est_rep = (
    df_chain[f"c_sex[{i_sex}]"].mean()
    + df_chain[f"b_sex[{i_sex}]"].mean() * x_graph.reshape(-1, 1).T
    + df_chain[f"a_sex[{i_sex}]"].mean()
    * (x_graph.reshape(-1, 1) * x_graph.reshape(-1, 1)).T
).reshape(-1)

az.plot_hdi(x_graph, y_graph, color="k", plot_kwargs={"ls": "--"})
plt.plot(x_graph, y_est_rep, "C6")


# %%
df = fit.to_dataframe()
# df.describe()
df
# df[[f"mu[{i+1}]" for i in range(10)]].mean()
# dir(fit)


# %%

