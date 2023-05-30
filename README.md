# Are They What They Claim: A Comprehensive Study of Ordinary Linear Regression Among the Top Machine Learning Libraries in Python
### by Sam Johnson, Josh Elms, Madhavan K R, Keerthana Sugasi, Parichit Sharma, Hasan Kurban and Mehmet M. Dalkilic
---
This repository was created to display supplementary materials from the above-mentioned \[[paper]()\] submitted to KDD2023. Below are steps to replicate the author's experiments in the paper. 

## Setup
To set up for a run of this pipeline, you will need to download and install the necessary libraries. Please ensure you have a python >= 3.7. Determine whether you are using pip or conda (if you don't know, use the instructions for pip). 

Pip users should run:
```
pip install -r requirements.txt
```

Conda users should run:
```
conda create --name <env> --file environment.yaml
conda activate <env>
```


### To recreate 'Runtime Comparison' Experiment

##### Steps:
---

### To recreate 'Memory Comparison' Experiment

##### Steps:
---

### To recreate 'Subsections of Circular Data' Experiment

##### Steps:

1. Run `circular_data_exper/data/create_data.py`
2. Run `circular_data_exper/analysis/run_lin_reg.py`
3. Run `circular_data_exper/analysis/aggregate_results.py`
4. Your result CSVs will be `circular_data_exper/analysis/final_results` folder and the their accompanying images will be in `circular_data_exper/analysis/regression_pics`.

To run again, delete `circular_data_exper/analysis/final_results`, `circular_data_exper/analysis/outputs`, `circular_data_exper/data/raw_data`, `circular_data_exper/analysis/regression_pics` folders and `circular_data_exper/analysis/cnt_#.txt` file. Start again with Step 1.
---

### To recreate 'High-Dimensional Data' Experiment

##### Steps:

1. Run `high_dimensional_exper/analysis/run_datasets.py`
2. Run `high_dimensional_exper/analysis/result_aggregation.py`

The results of this experiment are already stored in the `high_dimensional_exper/results` folder. The final CSV used in the paper is `high_dimensional_exper/results/MAE_linreg_comparison.csv`.

---

Email Joshua Elms (joshua.elms111@gmail.com) for questions.

If you find this work useful, cite it using:
```
@article{elms2023ares,
  title={Are They What They Claim: A Comprehensive Study of Ordinary Linear Regression Among the Top Machine Learning Libraries in Python},
  author={Johnson, Sam and Elms, Josh and Kalkunte Ramachandra, Madhavan and Sugasi, Keerthana and Sharma, Parichit, and Kurban, Hasan and Dalkilic, Mehmet M.},
  year={2023}
}
```