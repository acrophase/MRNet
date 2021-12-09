# MultiRespDL

A Deep Learning based Multitasking model for estimation of Respiratory Rate from ECG and accelerometer.

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
    
## System Setup
        1. pip install requirement.
### Train the model
      Run new_testbench.py
### Generate the data files
      Run data_file_generator.py
### Evaluation
       [Download the models for different configurations](https://drive.google.com/drive/folders/1wsyNcdeR1MF__zN9J5vhp9xQ8497aoV1?usp=sharing)
      
           
      
          



