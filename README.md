# MultiRespDL
[A Deep Learning Based Multitask Network for Respiration Rate Estimation - A Practical Perspective](https://arxiv.org/abs/2112.09071)

## Research

### Architecture
The architecture consist three blocks Encoder (F1), Decoder (F2), and IncResNet with Dense Layer (F3) as shown in figure below:
<p align="center">
  <image src = 'https://github.com/HTIC-HPOC/MultiRespDL/blob/main/plot/RespNet2_V2.0_block_crop.png'>
</p>
  
Different configuration using these blocks are designed as part of work. These configurations also differ in terms of inputs and outputs as given in the figure below:

<p align="center">
  <image src = 'https://github.com/HTIC-HPOC/MultiRespDL/blob/main/plot/Model_Table_6.0.png' >
</p>
  
### Datasets
  1. [PPG Dalia Dataset](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA)

### Quantitative Comparisons
  The comparison of the proposed model is done against the previously proposed works. The proposed model is also compared against the different configuration developed as a part 
  of work. The comparison is done in tems of Mean Square Error (MAE), Root Mean Square Error (RMSE), Parament Count (PC) and Inference time as shown in table below:
  <p align="center">
  <image src = 'https://github.com/HTIC-HPOC/MultiRespDL/blob/main/plot/Results.png' >
</p>
    
  The evaluation of model is also done during different activities, also to check the degree of agreement between the estimated RR and ground truth RR the box plot is used as     shown below:
    <p align="center">
    <image src = 'https://github.com/HTIC-HPOC/MultiRespDL/blob/main/plot/Plots_boc_ba.jpg' >
     </p>

## Repository Structure
```
.
├── Dayi_Bian
│   ├── CNN_EVAL.ipynb
│   ├── data_extraction.py
│   ├── data_file_generator.py
│   ├── filters.py
│   ├── hrv_analysis
│   │   ├── extract_features.py
│   │   ├── preprocessing.py
│   │   └── __pycache__
│   │       └── extract_features.cpython-38.pyc
│   ├── model.py
│   ├── new_testbench.py
│   ├── __pycache__
│   │   ├── data_extraction.cpython-38.pyc
│   │   ├── filters.cpython-38.pyc
│   │   ├── model.cpython-38.pyc
│   │   ├── resp_signal_extraction.cpython-38.pyc
│   │   └── rr_extration.cpython-38.pyc
│   ├── requirement.txt
│   ├── resp_signal_extraction.py
│   └── rr_extration.py
├── DL_Model
│   ├── data_extraction.py
│   ├── data_file_generator.py
│   ├── eval_testbench.ipynb
│   ├── filters.py
│   ├── hrv_analysis
│   │   ├── extract_features.py
│   │   └── preprocessing.py
│   ├── new_testbench.py
│   ├── requirement.txt
│   ├── resp_signal_extraction.py
│   ├── rr_extration.py
│   └── tf_model.py
├── LICENSE
├── plot
│   ├── activity_plot.png
│   ├── bland_altman.png
│   ├── Box_plot.png
│   ├── modality_plot.png
│   ├── Model_Table_6.0.png
│   ├── Plots_boc_ba.jpg
│   ├── RespNet2_V2.0_block_crop.png
│   └── Results.png
├── README.md
└── Smart_Fusion
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
## Acknowledgements

 - [hrv_analysis](https://github.com/neergaard/utime-pytorch)
 - [Tensorflow](https://github.com/tensorflow/tensorflow)
   
## Authors
- [Kapil Singh Rathore](https://github.com/Kapil19-dev)
---
      
**NOTE:**
- To run the specific method, open the corresponding folder and follow the steps.      
- Futhur modifications will be done in upcoming versions...      

---


      

      


      
           
      
          



