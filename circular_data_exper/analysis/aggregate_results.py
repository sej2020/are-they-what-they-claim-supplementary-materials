from yaml import load, dump, SafeLoader, SafeDumper
import numpy as np
from pathlib import Path
import pandas as pd
import imageio.v2 as iio

def construct_info_dict():
    input_path = Path("circular_data_exper/analysis/outputs")

    column_list = ["Partial Circle and its Regression Line", "$0^{\circ}$ Rotation", "$5^{\circ}$ Rotation", "$15^{\circ}$ Rotation", "$30^{\circ}$ Rotation", "$60^{\circ}$ Rotation", "$90^{\circ}$ Rotation"]
    df_3_subsets = pd.DataFrame(columns=column_list)
    df_4_subsets = pd.DataFrame(columns=column_list)
    df_5_subsets = pd.DataFrame(columns=column_list)
    df_0_subsets = pd.DataFrame(columns=column_list)

    for i, output_folder in enumerate([p for p in input_path.glob("*")]):
        results_path = output_folder / "results.yaml"
        metadata_path = output_folder / "metadata.yaml"
        image_path = output_folder / "regression.png"
        with open(results_path, "r") as f:
            results = load(f, SafeLoader)
            
        with open(metadata_path, "r") as f:
            metadata = load(f, SafeLoader)
        
        regression_pic = iio.imread(image_path)

        input_data_str = metadata["input_data"]
        splitted = input_data_str.split("_")
        n_subsets = splitted[1].split("-")[0]
        combo = splitted[2].split("-")[0]
        rot = splitted[3].split("-")[0]
        
        MAE = []
        for key in results.keys():
            MAE.append(results[key]["MAE"])
        MAE_av = round(np.mean(np.array(MAE)),3)
        MAE_std = np.std(np.array(MAE))
        if MAE_std > 0.0001:
            print("std too high")
        
        p = Path("circular_data_exper/analysis/regression_pics")
        p.mkdir(exist_ok=True, parents=True)

        if n_subsets == '3':
            if combo in list(df_3_subsets.index):
                df_3_subsets.loc[combo, f"${rot}^{{\circ}}$ Rotation"] = MAE_av
            else:
                row = {f"${rot}^{{\circ}}$ Rotation": MAE_av}
                df_3_subsets = pd.concat([df_3_subsets, pd.DataFrame(row,index=[combo])])
            if rot == '0':
                iio.imwrite(f"circular_data_exper/analysis/regression_pics/{n_subsets}-subsets_{combo}-combo.png", regression_pic)
                df_3_subsets.loc[combo, "Partial Circle and its Regression Line"] = f"{n_subsets}-subsets_{combo}-combo.png"

        if n_subsets == '4':
            if combo in list(df_4_subsets.index):
                df_4_subsets.loc[combo, f"${rot}^{{\circ}}$ Rotation"] = MAE_av
            else:
                row = {f"${rot}^{{\circ}}$ Rotation": MAE_av}
                df_4_subsets = pd.concat([df_4_subsets, pd.DataFrame(row,index=[combo])])
            if rot == '0':
                iio.imwrite(f"circular_data_exper/analysis/regression_pics/{n_subsets}-subsets_{combo}-combo.png", regression_pic)
                df_4_subsets.loc[combo, "Partial Circle and its Regression Line"] = f"{n_subsets}-subsets_{combo}-combo.png"

        if n_subsets == '5':
            if combo in list(df_5_subsets.index):
                df_5_subsets.loc[combo, f"${rot}^{{\circ}}$ Rotation"] = MAE_av
            else:
                row = {f"${rot}^{{\circ}}$ Rotation": MAE_av}
                df_5_subsets = pd.concat([df_5_subsets, pd.DataFrame(row,index=[combo])])
            if rot == '0':
                iio.imwrite(f"circular_data_exper/analysis/regression_pics/{n_subsets}-subsets_{combo}-combo.png", regression_pic)
                df_5_subsets.loc[combo, "Partial Circle and its Regression Line"] = f"{n_subsets}-subsets_{combo}-combo.png"

        if n_subsets == '0':
            row = {f"${rot}^{{\circ}}$ Rotation": MAE_av for rot in [0, 5, 15, 30, 60, 90]}
            df_0_subsets = pd.concat([df_0_subsets, pd.DataFrame(row,index=[combo])])
            if rot == '0':
                iio.imwrite(f"circular_data_exper/analysis/regression_pics/{n_subsets}-subsets_{combo}-combo.png", regression_pic)
                df_0_subsets.loc[combo, "Partial Circle and its Regression Line"] = f"{n_subsets}-subsets_{combo}-combo.png"            


    final_results = Path("circular_data_exper/analysis/final_results")
    final_results.mkdir(exist_ok=True, parents=True)
    for df, name in [(df_0_subsets, "0-subsets"), (df_3_subsets, "3-subsets"), (df_4_subsets, "4-subsets"), (df_5_subsets, "5-subsets")]:
        df.to_csv(f"circular_data_exper/analysis/final_results/{name}.csv", index=False)

    pass

construct_info_dict()
