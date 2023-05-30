from sklearn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# import torch
# import mxnet as mx
import pyaml
from pathlib import Path
from time import perf_counter, process_time


def linreg_pipeline(data_path: str, include_regs="all", split_pcnt=None, random_seed=None, time_type="total", 
                    vis_theme="whitegrid", output_folder=os.getcwd(), verbose_output=True, want_figs=True) -> dict:

    """
    This function is the main entry point for the linear regression pipeline. It takes in a path to a csv file, then performs
    linear regression using each of the tested OLS implementations. It then produces two yaml files and an image. The 
    first yaml file contains the results of the regression, the second contains the metadata of the run, and the image file is
    a plot of the results of the regression. These files are saved in an output folder.
    
    Args: 

        data_path (str): path to file that can become a pd.DataFrame or np.ndarray with target variable in final column 
                        and no categorical or missing data
        
        include_regs (str or container): "all" to use all algorithms or a list of desired algorithms to use a subset
                                        options - "tf-necd" ::: "tf-cod" ::: "pytorch-qrcp" ::: "pytorch-qr" 
                                        ::: "pytorch-svd" ::: "pytorch-svddc" ::: "sklearn-svddc" ::: "mxnet-svddc"
            
        split_pcnt (str or float): None to train and test the algorithm over the entirety of the data or a real number from 1 - 100 
                                    to use that percentage of the data as a training set and test on the remainder
        
        random_seed (int): only used by sklearn.model_selection.train_test_split if split_pcnt != None
        
        time_type (str): "total" to use perf_counter and measure time in sleep, or "process" to measure only cpu time 
                        with process_time
                    
        vis_theme (str): "whitegrid" by default, or specify any one of the below options
                        options - "darkgrid" ::: "whitegrid" ::: "dark" ::: "white" ::: "ticks"
        
    Returns:

        results_dict (dict): a dictionary containing regression results
        
    """

    # Reading and splitting the data into train and test sets
    data = pd.read_csv(data_path, header=None).values
    data, fields = data_ingestion(data)
    timer = set_time_type(time_type)
    reg_names = decide_regressors(include_regs)
    X_train, X_test, y_train, y_test = split_data(data, split_pcnt, random_seed)
    
    # Running the regression loop
    results_dict = regression_loop(X_train, y_train, X_test, timer, reg_names, verbose_output)

    successful_regs = list(results_dict.keys())

    metric_lst = [
        ("MAE", metrics.mean_absolute_error),
        ("MSE", metrics.mean_squared_error),
        ("RMSE", lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred)**(1/2)),
        ("R2", metrics.r2_score),
    ]

    # Adding distance evaluations to results_dict
    results_dict = process_results(results_dict, y_test, metric_lst)

    # Generating figures and saving results
    run_number = get_and_increment_run_counter()
    output_folder = create_output_folder(run_number)
    if want_figs:
        generate_figures(results_dict, X_test, y_test, vis_theme, successful_regs, output_folder)
    
    metadata = {
        "input_data": data_path.name,
        "completed_regs": successful_regs,
        "split_percent": split_pcnt if split_pcnt else "No train/test split",
        "random_seed": random_seed,
        "timer_method": time_type,
        "dataset_shape": f"{data.shape[0]} x {data.shape[1]}",
    }
    
    dump_to_yaml(output_folder / "metadata.yaml", metadata, True)
    dump_to_yaml(output_folder / "results.yaml", results_dict, verbose_output)
    
    return results_dict
    

def data_ingestion(data: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list]:
    """
    This function takes in a pd.DataFrame or np.ndarray and returns a np.ndarray and a list of column names.

    Args:
    
        data (pd.DataFrame or np.ndarray): data to be used in regression
    
    Returns:
        
        data (np.ndarray): data to be used in regression

        fields (list): list of column names
    """

    assert isinstance(data, (pd.DataFrame, np.ndarray)), f"Data must be of type pd.DataFrame or np.ndarray, not {type(data)}"
    fields = []
    if isinstance(data, pd.DataFrame):
        fields = data.columns
        data = data.values
    assert data.dtype != "object", "Data must be numeric, not object type - remove categorical data, impute missing values, or make array regular (not ragged)"

    return data, fields


def process_results(results_dict: dict, y_test: np.ndarray, metrics: list) -> dict:
    """
    This function takes in a dictionary of regression results and adds distance metrics to the dictionary.
    
    Args:
    
        results_dict (dict): dictionary of regression results
        
        y_test (np.ndarray): array of target values
        
        metrics (list): list of tuples containing metric names and metric functions
        
    Returns:
    
        results_dict (dict): dictionary of regression results with distance metrics added
    """

    for reg_name, reg_output in results_dict.items():
        for metric, formula in metrics:
            score = formula(y_test, reg_output["y_pred"])
            results_dict[reg_name][metric] = score
    return results_dict


def set_time_type(time_type: str) -> object:
    """
    This function takes in a string and returns a timer function based on the string.
    
    Args:
    
        time_type (str): "total" to use perf_counter and measure time in sleep, or "process" to measure only cpu time

    Returns:

        timer (function): a timer function
    """

    match time_type:
        case "total":
            timer = perf_counter
            
        case "process":
            timer = process_time
            
        case _:
            raise ValueError(f"time_type must be one of the options shown in the docs, not: {time_type}")
        
    return timer


def decide_regressors(include_regs: str | list | set | tuple) -> list:
    """
    This function takes in a string "all" or container and returns a list of regressors to use in the regression loop.
    
    Args:
    
        include_regs (str or container): "all" to use all algorithms or a list of desired algorithms to use a subset
        
    Returns:
    
        reg_names (list): list of regressors to use in the regression loop
    """
    possible_regressors = [
        "tf-necd",
        "tf-cod",
        "pytorch-qrcp",
        "pytorch-qr",
        "pytorch-svd",
        "pytorch-svddc",
        "sklearn-svddc",
        "mxnet-svddc",
    ]
    
    if include_regs == "all":
        reg_names = possible_regressors
        
    elif isinstance(include_regs, (tuple, list, set)):
        reg_names = list({reg_name for reg_name in include_regs if reg_name in possible_regressors})
        
    else: 
        raise ValueError(f'Invalid value passed for include_regs: {include_regs}\nSee documentation')
    
    return reg_names


def split_data(data: np.ndarray, split_pcnt: None | float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function takes in a numpy array and returns a train/test split of the data based on a specified split percentage.
    
    Args:
    
        data (np.ndarray): data to be split into train/test
        
        split_pcnt (int or float): percentage of data to be used for training
        
        seed (int): random seed to be used for reproducibility
        
    Returns:
    
        X_train (np.ndarray): training data

        X_test (np.ndarray): testing data

        y_train (np.ndarray): training labels

        y_test (np.ndarray): testing labels
    """
    assert isinstance(split_pcnt, (int, float)) or split_pcnt is None, f"Invalid value passed for split_pcnt: {split_pcnt}\nSee documentation"
    if split_pcnt is None:
        X_train, X_test, y_train, y_test = data[:, :-1], data[:, :-1], data[:, -1], data[:, -1]
        
    else:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(data, train_size = (split_pcnt / 100), random_state=seed)
        
    return X_train, X_test, y_train, y_test


def regression_loop(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, timer: object, reg_names: list, verbose_output: bool):
    """
    This function takes in training and testing data, and performs linear regression using each of the specified
     OLS implementations. It returns a dictionary of results including the trained model, the time to train the model,
     and the predictions.
    
    Args:
    
        X_train (np.ndarray): training data
        
        y_train (np.ndarray): training labels
        
        X_test (np.ndarray): testing data
        
        timer (function): a timer function
        
        reg_names (list): list of regressors to use in the regression loop
        
        verbose_output (bool): whether to include the model in the results dictionary
        
    Returns:
    
        results_dict (dict): dictionary of results
    """

    results_dict = {}
        
    for reg_name in reg_names:       

        start_lstsq = timer()
        match reg_name:
            case "sklearn-svddc":
                model = linear_model.LinearRegression(fit_intercept=False).fit(X_train,y_train).coef_

            case "tf-necd":
                model = tf.linalg.lstsq(X_train, y_train[...,np.newaxis], fast=True).numpy()
                
            case "tf-cod":
                model = tf.linalg.lstsq(X_train, y_train[...,np.newaxis], fast=False).numpy()

            case "pytorch-qrcp":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gelsy").solution)

            case "pytorch-qr":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gels").solution)

            case "pytorch-svd":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gelss").solution)

            case "pytorch-svddc":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gelsd").solution)

            case "mxnet-svddc":
                model = mx.np.linalg.lstsq(X_train, y_train[...,np.newaxis], rcond=None)[0]
            
        pred = X_test @ model 
        
        stop_lstsq = timer()

        results_dict[reg_name] = {
            "elapsed_time": stop_lstsq - start_lstsq,
            "y_pred": pred
            }
        
        if verbose_output:
            results_dict[reg_name]["model"] = model
        
    return results_dict


def dump_to_yaml(path: Path, object: dict, verbose_output = True):
    """
    This function takes in a dictionary of results and dumps it to a yaml file.
    
    Args:
    
        path (str): path to the output file
        
        object (dict): dictionary to be written to yaml
        
        verbose_output (bool): whether to include the model predictions in the yaml file
        
    Returns:
    
        None
    """
    if not verbose_output:
        for reg in object.keys():
            del object[reg]["y_pred"]
    
    with open(path, "w") as f_log:
        dump = pyaml.dump(object)
        f_log.write(dump)


def generate_figures(results_dict: dict, X_test: np.ndarray, y_test: np.ndarray, vis_theme: str, successful_regs: list,
                      output_folder: Path):
    """
    This function creates a figure depicting the circular data and its regression line and saves the images in the output folder

    Args:

        results_dict (dict): dictionary of regression results

        X_test (np.ndarray): testing data

        y_test (np.ndarray): testing labels

        vis_theme (str): visualization theme

        successful_regs (list): list of regressors that were successfully run

        output_folder (str): path to the output folder

    Returns:

        None
    """

    # Styling the plots
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    CHONK_SIZE = 24
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:black")
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black") #  powder blue
        
    possible_themes = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
    assert vis_theme in possible_themes, f"Invalid value passed for vis_theme: {vis_theme}\nSee documentation"
    sns.set_style(vis_theme, {'font.family':['serif'], 'axes.edgecolor':'black','ytick.left': True})
    plt.ticklabel_format(style = 'plain')
      
    fig, ax = plt.subplots()

    # Plotting the data points
    sns.scatterplot(x=X_test.flatten(), y=y_test.flatten(), ax=ax, color="blue", edgecolor="blue", s=100)

    # To produce regression line on the interval bounded by -50 and 50
    X_range = np.linspace(-50, 50, 2)[:, np.newaxis]

    # Plotting the regression line
    reg_lines = [X_range @ results_dict[regressor]["model"] for regressor in successful_regs]
    for line, regressor in zip(reg_lines, successful_regs):
        ax.plot(X_range.flatten(), line.flatten(), color='black', alpha = 0.75, linewidth=8)
    
    # Plotting a thin line over x-axis and y-axis
    ax.plot([i for i in range(-50,50)], [0 for _ in range(-50,50)], linestyle="dashed", color="gray", alpha=0.5)
    ax.plot([0 for _ in range(-50,50)], [i for i in range(-50,50)], linestyle="dashed", color="gray", alpha=0.5)

    ax.set_ylim(-12,12)
    ax.set_xlim(-15,15)
    ax.grid(False)
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    plt.savefig(output_folder / f"regression.png", dpi=300, bbox_inches='tight', pad_inches=0.0)
        
    plt.clf()
    plt.close(fig="all")


def get_and_increment_run_counter() -> int:
    """
    This function is used to keep track of the number of times the program has been run, by incrementing the name
    of a file called "cnt" by 1 each time the program is run.

    Args:

        None

    Returns:

        cnt (int): The number of times the program has been run
    """
    program_container = list(Path.cwd().rglob("run_lin_reg.py"))[0].parent
    cnt_file_lst = list(program_container.glob("cnt_*"))
    
    if not cnt_file_lst:
        cnt_file = program_container / "cnt_1"
        cnt_file.touch()
    else:
        cnt_file = cnt_file_lst[0]
        
    cnt = int(cnt_file.stem.split("_")[-1])
    cnt_file.rename(cnt_file.parent / f"cnt_{cnt+1}")
    
    return cnt
    

def create_output_folder(run_number: int) -> Path:
    """
    This function creates a folder to store the outputs of the program. The folder is named "output_{run_number}".
    
    Args:
    
        run_number (int): The number of times the program has been run
        
    Returns:
    
        output_folder (Path): The path to the folder where the outputs will be stored
    """
    program_container = list(Path.cwd().rglob("run_lin_reg.py"))[0].parent
    output_folder = program_container /"outputs" / f"output_{run_number}"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    return output_folder
    

def main(data_path: str, params: dict):
    
    linreg_pipeline(data_path, **params)
        
    print("Run complete")


if __name__ == "__main__":
    container_path = Path("circular_data_exper/data/raw_data")
    
    for hyper_path in container_path.glob("_*"):
        main(
            data_path = hyper_path,
            params = {
                "random_seed": 100,
                "include_regs": ["sklearn-svddc"]
            }
        )
    

