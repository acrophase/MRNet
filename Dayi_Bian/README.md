## STRUCTURE OF DAYI_BIAN
```
.
├── CNN_EVAL.ipynb
├── data_extraction.py
├── data_file_generator.py
├── filters.py
├── hrv_analysis
│   ├── extract_features.py
│   ├── preprocessing.py
│   └── __pycache__
│       └── extract_features.cpython-38.pyc
├── model.py
├── new_testbench.py
├── __pycache__
│   ├── data_extraction.cpython-38.pyc
│   ├── filters.cpython-38.pyc
│   ├── model.cpython-38.pyc
│   ├── resp_signal_extraction.cpython-38.pyc
│   └── rr_extration.cpython-38.pyc
├── requirement.txt
├── resp_signal_extraction.py
└── rr_extration.py
```
## System Setup for DAYI_BIAN
        1. pip install requirement.

## Steps for DAYI_BIAN
  -- General Steps
  * Download the dataset.
  -- Generate the data files.
  * Run data_file_generator.py
  
  -- Evaluate the model.
  * [Download the models for different configurations](https://drive.google.com/drive/folders/1fEA6SkJ1m2DwxU-OqSed1VjLRy3Q10dJ?usp=sharing)
  * [Download the data files](https://drive.google.com/drive/folders/1PIaNOR3ddFgQ0L0QK-3v3PvOojIceIZ3?usp=sharing)
  * CNN_EVAL.ipynb
