from pathlib import Path
import glob
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from statsmodels.stats.proportion import proportion_confint

### FREQUENTIST POWER CALCULATION

# def compute_exact_power(N, true_b1=0.1, true_sigma=0.1, alpha=0.025):
#     df = N - 2
#
#     sd_x = 0.2887 # sd of x in U(0,1)
#
#     # Standard Error of slope & Non-Centrality Parameter
#     se_b1 = true_sigma / (sd_x * np.sqrt(N))
#     ncp = true_b1 / se_b1
#
#     # One-sided critical value
#     t_crit = stats.t.ppf(1 - alpha, df=df)
#
#     # Power
#     return 1 - stats.nct.cdf(t_crit, df=df, nc=ncp)
#
# for ss in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]:
#     print(f"ss={ss} and power is {compute_exact_power(ss)}")

UNI_COLORS = ["#0065BD", "#E37222", "#A2AD00"]

TUM_COLORS = {
    "BayesFlow": "#0065BD",
    "PyMC": "#E37222",
    "Bayesian": "#E37222",
    "Frequentist": "#000000"
}

### COMPARISON OF FREQUENTIST AND BAYESIAN POWER CURVES

df_f = pd.read_excel("../Results/output_1_base_FREQ.xlsx")
df_p = pd.read_excel("../Results/output_combined.xlsx")[["Scenario", "NameOfModel", "EstimatedPower", "SampleSize"]]
df_p = df_p[(df_p["Scenario"] == "1_base") & (df_p["NameOfModel"] == "SSP_BayesFlow_1_base_PyMC")]
df_p["NameOfModel"] = "Bayesian"

df = pd.concat([df_f, df_p])
df.rename(columns={"NameOfModel": "Model", "EstimatedPower": "Estimated Power", "SampleSize": "Candidate Sample Size"}, inplace=True)

NumberOfSims = 500

df["lower"], df["upper"] = proportion_confint(count=df["Estimated Power"] * NumberOfSims,
                                              nobs=NumberOfSims, alpha=0.05, method='wilson')
df.loc[df["Model"] == "Frequentist", "lower"] = np.nan
df.loc[df["Model"] == "Frequentist", "upper"] = np.nan

(
    ggplot(df, aes("Candidate Sample Size", "Estimated Power", color="Model"))
    + geom_point()
    + geom_line()
    + geom_ribbon(aes(ymin="lower", ymax="upper", fill="Model"), alpha=0.15, color="None")
    + geom_hline(yintercept=[0.8], color="black", linetype="dashed")
    + scale_color_manual(values=TUM_COLORS)
    + scale_fill_manual(values=TUM_COLORS)
    + theme(figure_size=(6, 4))
).show()

### CODE FOR POWER CURVES, WILSON INTERVALS, AND RMSE-VALUES

df = pd.read_excel("../Results/output_combined.xlsx")[["Scenario", "NameOfModel", "EstimatedPower", "SampleSize"]]
df["Scenario"] = ["1. Base" if str(sc).startswith("1")
                     else "2. Weak Signal" if str(sc).startswith("2")
                     else "3. Uninformative" if str(sc).startswith("3")
                     else "4. Narrow Prior" for sc in df["Scenario"]]
df["NameOfModel"] = ["BayesFlow" if str(m).endswith("F")
                     else "PyMC" for m in df["NameOfModel"]]

df.rename(columns={"NameOfModel": "Model", "EstimatedPower": "Estimated Power", "SampleSize": "Candidate Sample Size"},
          inplace=True)

NumberOfSims = 500

df["lower"], df["upper"] = proportion_confint(count=df["Estimated Power"] * NumberOfSims,
                                              nobs=NumberOfSims, alpha=0.05, method='wilson')

(
    ggplot(df, aes("Candidate Sample Size", "Estimated Power", color="Model"))
    + geom_point()
    + geom_line()
    + geom_ribbon(aes(ymin="lower", ymax="upper", fill="Model"), alpha=0.15, color="None")
    + geom_hline(yintercept=[0.8], color="black", linetype="dashed")
    + facet_wrap("Scenario", scales="free_x", nrow=1)
    + scale_color_manual(values=TUM_COLORS)
    + scale_fill_manual(values=TUM_COLORS)
    + theme(figure_size=(12, 4))
).show()

df_pivot = pd.pivot(df, index=["Candidate Sample Size", "Scenario"], columns="Model", values="Estimated Power")
df_pivot["sq_diff"] = (df_pivot["PyMC"] - df_pivot["BayesFlow"]) ** 2
rmses = np.sqrt(df_pivot.groupby("Scenario")["sq_diff"].mean())
print(rmses)

### CODE FOR RUNTIME COMPARISONS

df = pd.read_excel("../Results/runtimes.xlsx")
df["Scenario"] = ["1. Base" if str(sc).startswith("1")
                     else "2. Weak Signal" if str(sc).startswith("2")
                     else "3. Uninformative" if str(sc).startswith("3")
                     else "4. Narrow Prior" for sc in df["Scenario"]]
df["Model"] = ["BayesFlow" if str(m).endswith("F")
                     else "PyMC" for m in df["Model"]]

(
    ggplot(df, aes("K", "Time (s)", color="Model"))
    + geom_point()
    + geom_line()
    + facet_wrap("Scenario", scales="free_x", nrow=1)
    + scale_color_manual(values=TUM_COLORS)
    + theme(figure_size=(12, 4))
).show()

### NUMBER OF SIMS: POWER CURVES

df = pd.read_excel("../Results/num_of_sims_comparison.xlsx")[["NameOfModel", "EstimatedPower", "SampleSize", "NumberOfSims"]]
df["NameOfModel"] = ["BayesFlow" if str(m).endswith("F")
                     else "PyMC" for m in df["NameOfModel"]]

df.rename(columns={"NameOfModel": "Model", "EstimatedPower": "Estimated Power", "SampleSize": "Candidate Sample Size",
                   "NumberOfSims": "M"}, inplace=True)

df["lower"], df["upper"] = proportion_confint(count=df["Estimated Power"] * df["M"],
                                              nobs=df["M"], alpha=0.05, method='wilson')

(
    ggplot(df, aes("Candidate Sample Size", "Estimated Power", color="Model"))
    + geom_point()
    + geom_line()
    + geom_ribbon(aes(ymin="lower", ymax="upper", fill="Model"), alpha=0.15, color="None")
    + geom_hline(yintercept=[0.8], color="black", linetype="dashed")
    + facet_wrap("M", scales="free_x", nrow=1)
    + scale_color_manual(values=TUM_COLORS)
    + scale_fill_manual(values=TUM_COLORS)
    + theme(figure_size=(10, 4))
).show()

### NUMBER OF SIMS: RUNTIMES

df = pd.read_excel("../Results/num_of_sims_runtimes.xlsx")[["NameOfModel", "InferenceRuntime", "TotalRuntime", "NumberOfSims"]]
df["NameOfModel"] = ["BayesFlow" if str(m).endswith("F")
                     else "PyMC" for m in df["NameOfModel"]]

df.rename(columns={"NameOfModel": "Model", "InferenceRuntime": "Inference Runtime", "TotalRuntime": "Total Runtime", "NumberOfSims": "M"},
          inplace=True)

(
    ggplot(df, aes("Model", "Total Runtime", color="Model"))
    + geom_col(aes(fill="Model"))
    + facet_wrap("M", scales="free_x", nrow=1)
    + scale_color_manual(values=TUM_COLORS)
    + scale_fill_manual(values=TUM_COLORS)
    + theme(figure_size=(10, 4))
).show()

### SCENARIO B_MISSPECIFIED: POWER CURVES

df = pd.read_excel("../Results/output_SSP_BayesFlow_B_misspecified.xlsx")[["Scenario", "NameOfModel", "EstimatedPower", "SampleSize"]]

df["NameOfModel"] = ["BayesFlow" if str(m).endswith("F")
                     else "PyMC" for m in df["NameOfModel"]]

df.rename(columns={"NameOfModel": "Model", "EstimatedPower": "Estimated Power", "SampleSize": "Candidate Sample Size"},
          inplace=True)

NumberOfSims = 500

df["lower"], df["upper"] = proportion_confint(count=df["Estimated Power"] * NumberOfSims,
                                              nobs=NumberOfSims, alpha=0.05, method='wilson')
df.loc[df["Model"] == "Frequentist", "lower"] = np.nan
df.loc[df["Model"] == "Frequentist", "upper"] = np.nan

(
    ggplot(df, aes("Candidate Sample Size", "Estimated Power", color="Model"))
    + geom_point()
    + geom_line()
    + geom_ribbon(aes(ymin="lower", ymax="upper", fill="Model"), alpha=0.15, color="None")
    + geom_hline(yintercept=[0.8], color="black", linetype="dashed")
    + scale_color_manual(values=TUM_COLORS)
    + scale_fill_manual(values=TUM_COLORS)
    + theme(figure_size=(6, 4))
).show()

