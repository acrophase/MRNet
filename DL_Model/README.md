## Structure for DL_method
```
├── data_extraction.py
├── data_file_generator.py
├── eval_testbench.ipynb
├── filters.py
├── hrv_analysis
│   ├── extract_features.py
│   └── preprocessing.py
├── new_testbench.py
├── requirement.txt
├── resp_signal_extraction.py
├── rr_extration.py
└── tf_model.py
```
      
## System Setup for DL_Method
        1. pip install requirement.txt.

## Steps for DL_Method
  -- General Steps
  * Download the dataset.
      
  -- Generate the data files.
  ```bash
  python data_file_generator.py --data_path <"Path of ppg dalia data"> --srate <"sampling rate"> --win_len <"window length"> --num_epochs <"number of epochs">
  ```
  
  -- Train the model.
  * Run new_testbench.py
  
  -- Evaluate the model.
  * [Download the models for different configurations](https://drive.google.com/drive/folders/1wsyNcdeR1MF__zN9J5vhp9xQ8497aoV1?usp=sharing)
  * [Download the data files](https://drive.google.com/drive/folders/1PIaNOR3ddFgQ0L0QK-3v3PvOojIceIZ3?usp=sharing)
  * Run eval_testbench.ipynb
