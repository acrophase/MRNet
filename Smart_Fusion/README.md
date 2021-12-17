## Structure for DL_method

```
.
├── edr_adr_signal_extraction.py
├── extract_features.py
├── filters.py
├── hrv_analysis
│   ├── extract_features.py
│   └── preprocessing.py
├── machine_learning.py
├── plots.py
├── ppg_dalia_data_extraction.py
├── preprocessing.py
├── Ref_signal_Testbench.ipynb
├── Respiratory_signal_plot_testbench .ipynb
├── rqi_extraction.py
├── rr_extraction.py
├── testbench.py
└── validation.py

```
## System Setup for Smart_Fusion
        1. pip install requirement.txt.

## Steps for DL_Method

  -- General Steps
  * Download the dataset.
  
  -- Training and Evaluation.
  ```bash
  python testbench.py --data_path <"Path of ppg dalia data">
  ```
