from sklearn import *
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# import torch
# import mxnet as mx

def read_data(data_path: str) -> np.ndarray:
    """
    Reads in data from a csv file and returns a numpy array of the data

    Args:

        data_path (str) - path to the csv file

    Returns:

        arr (nd.array) - numpy array of the data
    """
    df = pd.read_csv(data_path, na_values="?")
    df = df.fillna(0)
    arr = df.values
    return arr[:, :-1], arr[:, -1]


def gen_cv_samples(X_train: np.ndarray, y_train: np.ndarray, n_cv_folds: int) -> list:
    """
    Generates a nested array of length k (where k is the number of cv folds)
    Each sub-tuple contains k folds formed into training data and the k+1 fold left out as test data
    
    Args: 

        X_train (nd.array) - training data already processed

        y_train (nd.array) - training labels already processed

        n_cv_folds (int) - number of cv folds to generate
        
    Returns: 

        train/test data (tuples) - nested_samples gets broken down into four list
    """
    kf = model_selection.KFold(n_splits = n_cv_folds, shuffle = True, random_state = 100)
    kf_indices = [(train, test) for train, test in kf.split(X_train, y_train)]
    nested_samples = [(X_train[train_idxs], y_train[train_idxs], X_train[test_idxs], y_train[test_idxs]) for train_idxs, test_idxs in kf_indices]
    return nested_samples
    

def run_linreg(cv_data, regr_name, formula):
    """
    This function runs the linear regression models on the data and returns the results of some error metric

    Args:

        cv_data (tuples) - nested array of training and test data

        regr_name (str) - name of the regression model to run

        formula (function) - error metric to use

    Returns:

        accumulator (list) - list of error metrics for each fold

        error (list) - list of errors that occured during the run
    """
    accumulator = []
    error = []
    try:
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(cv_data):
            pred = None
            match regr_name:
                case "sklearn-svddc":
                    model = linear_model.LinearRegression(fit_intercept=False).fit(X_tr,y_tr).coef_

                case "tf-necd":
                    model = tf.linalg.lstsq(X_tr, y_tr[...,np.newaxis], fast=True).numpy()
                    
                case "tf-cod":
                    model = tf.linalg.lstsq(X_tr, y_tr[...,np.newaxis], fast=False).numpy()

                case "pytorch-qrcp":
                    model = np.array(torch.linalg.lstsq(torch.Tensor(X_tr), torch.Tensor(y_tr[...,np.newaxis]), driver="gelsy").solution)

                case "pytorch-qr":
                    model = np.array(torch.linalg.lstsq(torch.Tensor(X_tr), torch.Tensor(y_tr[...,np.newaxis]), driver="gels").solution)

                case "pytorch-svd":
                    model = np.array(torch.linalg.lstsq(torch.Tensor(X_tr), torch.Tensor(y_tr[...,np.newaxis]), driver="gelss").solution)

                case "pytorch-svddc":
                    model = np.array(torch.linalg.lstsq(torch.Tensor(X_tr), torch.Tensor(y_tr[...,np.newaxis]), driver="gelsd").solution)

                case "mxnet-svddc":
                    model = mx.np.linalg.lstsq(X_tr, y_tr[...,np.newaxis], rcond=None)[0]
                    
            pred = X_te @ model 

            results = formula(y_te, pred)
        
            accumulator.append(results)

    except Exception as e:
        error.append(e)
        return None, error
            
    return accumulator, error

    
def main(data_path, k_folds, data_name, reg_names):
    """
    This is the pipeline to read data, run regression on OLS implementations, and save the results. The results will
    be saved as a CSV and text file for each regressor for each error metric.
    
    Args:
    
        data_path (str) - path to the csv file
        
        k_folds (int) - number of cv folds to generate
        
        data_name (str) - name of the dataset
        
        reg_names (list) - list of regression names to run
        
    Returns:
    
        csv files of results
    """
    X, y = read_data(data_path)
    cv_data = gen_cv_samples(X, y, k_folds)
    
    metric_lst = [
        ("MAE", metrics.mean_absolute_error),
        ("MSE", metrics.mean_squared_error),
        ("RMSE", lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred)**(1/2)),
        ("R2", metrics.r2_score),
    ]

    # run the regression models and recording error metrics and thrown errors for each model
    for metric_name, formula in metric_lst:
        result_accumulator = {}
        err_accumulator = {}
        for name in reg_names:
            res, err = run_linreg(cv_data, name, formula)
            if res:
                result_accumulator[name] = res
            if err:
                err_accumulator[name] = err
        
        results_df = pd.DataFrame(result_accumulator)
        results_df.to_csv(f"high_dimensional_exper/data/results/{data_name}-{metric_name}_linreg_comparison.csv")
    
        with open(f"high_dimensional_exper/data/results/{data_name}-{metric_name}_errors.err", "a") as e_log:
            for k, v in err_accumulator.items():
                e_log.write(f"{k}: {v}\n")


if __name__ == "__main__":
    high_dim_data = {"Superconductivity": "high_dimensional_exper/data/Conductivity.csv",
                        "Residential Building": "BetaDataExper/HighDimData/data/Residential-Building-Data-Set.csv",
                        "Geographical Origin of Music": "BetaDataExper/HighDimData/data/Geographical-Music.csv",
                        "Facebook Comment Volume": "BetaDataExper/HighDimData/data/Facebook-Comments.csv",
                        "Online News Popularity": "BetaDataExper/HighDimData/data/Online-News-Popularity.csv",                
                        "Communities and Crime": "BetaDataExper/HighDimData/data/communities.csv",
                        "Hailstone": "BetaDataExper/HighDimData/data/hailstone_data.csv",
                        "Hourly Energy Demand": "BetaDataExper/HighDimData/data/energy_dataset.csv",
                        "KEGG Metabolic Pathway": "BetaDataExper/HighDimData/data/KEGG-Metabolic.csv",
                        "Blog Feedback": "BetaDataExper/HighDimData/data/blog.csv"}
    reg_names = ["tf-necd", "sklearn-svddc"]
              #  "tf-necd", "tf-cod", "pytorch-qrcp", "pytorch-qr", "pytorch-svd", "pytorch-svddc", "sklearn-svddc", "mxnet-svddc"
    for data_name, path in high_dim_data.items():
        main(data_path = path, k_folds = 10, data_name = data_name, reg_names = reg_names)
        break
        