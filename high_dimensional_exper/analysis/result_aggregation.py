import pandas as pd
import numpy as np

blog = pd.read_csv("high_dimensional_exper/data/results/Blog Feedback-MAE_linreg_comparison.csv", index_col=0)
communities = pd.read_csv("high_dimensional_exper/data/results/Communities and Crime-MAE_linreg_comparison.csv", index_col=0)
facebook = pd.read_csv("high_dimensional_exper/data/results/Facebook Comment Volume-MAE_linreg_comparison.csv", index_col=0)
geographical = pd.read_csv("high_dimensional_exper/data/results/Geographical Origin of Music-MAE_linreg_comparison.csv", index_col=0)
hailstone = pd.read_csv("high_dimensional_exper/data/results/Hailstone-MAE_linreg_comparison.csv", index_col=0)
energy = pd.read_csv("high_dimensional_exper/data/results/Hourly Energy Demand-MAE_linreg_comparison.csv", index_col=0)
kegg = pd.read_csv("high_dimensional_exper/data/results/KEGG Metabolic Pathway-MAE_linreg_comparison.csv", index_col=0)
online = pd.read_csv("high_dimensional_exper/data/results/Online News Popularity-MAE_linreg_comparison.csv", index_col=0)
residential = pd.read_csv("high_dimensional_exper/data/results/Residential Building-MAE_linreg_comparison.csv", index_col=0)
superconductivity = pd.read_csv("high_dimensional_exper/data/results/Superconductivity-MAE_linreg_comparison.csv", index_col=0)

final_df = pd.DataFrame(index = ["Blog Feedback", "Communities and Crime", "Facebook Comment Volume", "Geographical Origin of Music", "Hailstone", "Hourly Energy Demand", "KEGG Metabolic Pathway", "Online News Popularity", "Residential Building", "Superconductivity"],
                  columns = ["tf-necd", "tf-cod", "pytorch-qrcp", "pytorch-qr", "pytorch-svd", "pytorch-svddc", "sklearn-svddc", "mxnet-svddc"])
for df, idx in zip([blog, communities, facebook, geographical, hailstone, energy, kegg, online, residential, superconductivity], ["Blog Feedback", "Communities and Crime", "Facebook Comment Volume", "Geographical Origin of Music", "Hailstone", "Hourly Energy Demand", "KEGG Metabolic Pathway", "Online News Popularity", "Residential Building", "Superconductivity"]):
    for col in df.columns:
        final_df.loc[idx,col] = round(df.mean(axis=0)[col],3)

final_df.fillna("Failed", inplace=True)
print(final_df)
final_df.to_csv("high_dimensional_exper/data/results/MAE_linreg_comparison.csv")