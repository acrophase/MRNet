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
    
   #### Evaluation during different activities
    The evaluation of model is also done during different activities, as shown in the box plot below:
    <p align="center">
    <image src = 'https://github.com/HTIC-HPOC/MultiRespDL/blob/main/plot/activity_plot.png' >
     </p>
    
  
 


