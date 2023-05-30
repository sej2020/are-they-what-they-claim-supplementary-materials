# Are They What They Claim: A Comprehensive Study of Ordinary Linear Regression Among the Top Machine Learning Libraries in Python
### by Sam Johnson, Josh Elms, Madhavan K R, Keerthana Sugasi, Parichit Sharma, Hasan Kurban and Mehmet M. Dalkilic
---
This repository was created to display supplementary materials from the above-mentioned paper submitted to KDD2023. Below are steps to replicate experiments in the paper for interested readers. 

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

<p>To run again, delete `circular_data_exper/analysis/final_results`, `circular_data_exper/analysis/outputs`, `circular_data_exper/data/raw_data`, `circular_data_exper/analysis/regression_pics` folders and `circular_data_exper/analysis/cnt_#.txt` file. Start again with Step 1.</p>
***

### To recreate 'High-Dimensional Data' Experiment

##### Steps:
