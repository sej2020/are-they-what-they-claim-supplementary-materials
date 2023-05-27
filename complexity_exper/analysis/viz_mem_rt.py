import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from pathlib import Path

# Load data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def main(output_dir):
    # Load data
    input_data = output_dir / "processed_output"
    mem_df = load_data(input_data / "bytes.csv")
    act_rt_df = load_data(input_data / "actual_runtime.csv")
    theo_rt_df = load_data(input_data / "theoretical_runtime.csv")
    row_counts = list(mem_df.iloc[:,0])
    print(act_rt_df)
    print(row_counts)
    pass
    # Setup for plotting
    label_dict_rt ={
        "tf-necd": "TensorFlow (NE-CD)",
        "tf-cod": "TensorFlow (COD)",
        "pytorch-qrcp": "PyTorch (QRCP)",
        "pytorch-qr": "PyTorch (QR)",
        "pytorch-svd": "PyTorch (SVD)",
        "pytorch-svddc": "PyTorch (SVDDC)",
        "sklearn-svddc": "scikit-learn (SVDDC)",
    }

    label_dict_mem ={
        "tf-necd": "TensorFlow (NE-CD)",
        "tf-cod": "TensorFlow (COD)",
        "pytorch-qrcp": "All PyTorch solvers",
        "sklearn-svddc": "scikit-learn (SVDDC)",
    }

    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:black")
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black") #  powder blue
    plt.rc('lines', linewidth=2.5)
    plt.rc('grid', linewidth=1.3)

        
    sns.set_style("whitegrid", {'font.family':['serif'], 'axes.edgecolor':'black','ytick.left': True})

    # Plotting memory
    solvers = list(mem_df.columns[1:])
    fig, ax = plt.subplots()
    for solver in solvers:
        if solver in ['tf-necd', 'tf-cod', 'pytorch-qrcp','sklearn-svddc']:
            ax.plot(row_counts, mem_df[solver], label=label_dict_mem[solver])
    ax.set_xlabel("Number of rows in dataset")
    ax.set_ylabel("Memory usage (GB)")

    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    # ax.set_title("Memory usage of OLS solvers")
    labels = ["0", "25", "50", "75", "100", "125", "150", "175", "200", "225", "250"]
    ax.yaxis.set_major_locator(ticker.FixedLocator([i*0.25*10**11 for i in range(11)]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    plt.xscale("log")
    ax.legend()
    mem_figs_path = output_dir / "mem_figs"
    mem_figs_path.mkdir(exist_ok=True)
    plt.savefig(mem_figs_path / "all_memory.png", dpi=300, bbox_inches="tight")
    plt.clf()

    #Plotting small-scale memory
    solvers = list(mem_df.columns[1:])
    mem_df.drop(mem_df.tail(6).index, inplace = True)
    fig, ax = plt.subplots()
    for solver in solvers:
        if solver in ['tf-necd', 'tf-cod', 'pytorch-qrcp','sklearn-svddc']:
            ax.plot(row_counts[:-6], mem_df[solver], label=label_dict_mem[solver])
    ax.set_xlabel("Number of rows in dataset")
    ax.set_ylabel("Memory usage (bytes)")

    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    # ax.set_title("Memory usage of OLS solvers")
    labels = ["0", "25", "50", "75", "100", "125", "150", "175", "200", "225", "250"]
    ax.yaxis.set_major_locator(ticker.FixedLocator([i*0.25*10**8 for i in range(11)]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    plt.xscale("log")
    plt.yscale("log")
    ax.legend()
    plt.savefig(mem_figs_path / "small-scale_memory.png", dpi=300, bbox_inches="tight")
    plt.clf()


    # Plotting runtime
    act_rt_df_s = act_rt_df.iloc[:,1:].div(1e9)
    theo_rt_df_s = theo_rt_df.iloc[:,1:].div(1e9)
    rt_figs_path = output_dir / "rt_figs"
    rt_figs_path.mkdir(exist_ok=True)

    solvers = list(act_rt_df_s.columns)
    for solver, color in zip(solvers,["red", "darkblue", "darkgreen", "orange", "purple", "mediumvioletred", "slategray"]):
        fig, ax = plt.subplots()
        ax.plot(row_counts, act_rt_df_s[solver], label=label_dict_rt[solver]+" - Actual", color=color)
        ax.plot(row_counts, theo_rt_df_s[solver], label=label_dict_rt[solver]+" - Theoretical", color=color, linestyle="dashed")
        ax.set_xlabel("Number of rows in dataset")
        ax.set_ylabel("Runtime (s)")

        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)

        # ax.set_title("Runtime of OLS solvers")
        plt.xscale("log")
        ax.legend()
        plt.savefig(rt_figs_path / f"{solver}.png", dpi=300, bbox_inches="tight")
        plt.clf()

    # Plotting runtime - log
    act_rt_df_ms = act_rt_df.iloc[:,1:].div(1e6)
    theo_rt_df_ms = theo_rt_df.iloc[:,1:].div(1e6)

    solvers = list(act_rt_df_ms.columns)
    for solver, color in zip(solvers,["red", "darkblue", "darkgreen", "orange", "purple", "mediumvioletred", "slategray"]):
        fig, ax = plt.subplots()
        ax.plot(row_counts, act_rt_df_ms[solver], label=label_dict_rt[solver]+" - Actual", color=color)
        ax.plot(row_counts, theo_rt_df_ms[solver], label=label_dict_rt[solver]+" - Theoretical", color=color, linestyle="dashed")
        ax.set_xlabel("Number of rows in dataset")
        ax.set_ylabel("Runtime (ms)")

        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)

        # ax.set_title("Runtime of OLS solvers")
        plt.xscale("log")
        plt.yscale("log")
        ax.legend()
        plt.savefig(rt_figs_path / f"{solver}_log.png", dpi=600, bbox_inches="tight")
        plt.clf()




if __name__ == '__main__':
    output_dir = Path("BetaDataExper/BigOTest/postprocessing") / "quartz_run3"
    main(output_dir)
