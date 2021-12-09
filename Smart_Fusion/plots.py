import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

def box_plot_resp(error):
    '''
    input -- Absolute error in form of dataframe, based on different methods.
    output -- returns the boxplot array
    description -- This function takes the absolute error dataframe and then return
    the boxplot.
    '''
    df = error
    method = list(df.columns)
    boxplot = df.boxplot(column= method , vert = True , showfliers = False,grid = False)
    return boxplot

def bland_altman_plot(predicted , truth):
    '''
    Inputs -- predicted -- Predicted Data
              Truth -- Reference Data
    Output -- None
    Description -- Function gives the bland altman plot.
    '''
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    mean_val = np.mean([predicted , truth] , axis=0)
    diff = truth-predicted
    mean_diff = np.mean(diff)
    std_diff = np.std(diff , axis=0)

    plt.scatter(mean_val , diff)
    plt.axhline(mean_diff , color = 'black' , linestyle = '--' , linewidth = 3)
    plt.axhline(mean_diff + 1.96*std_diff , color = 'black' , linestyle = '--', linewidth = 3)
    plt.axhline(mean_diff - 1.96*std_diff , color = 'black' , linestyle = '--', linewidth = 3)
    plt.xlabel('Average RR')
    plt.ylabel('RR_fused -  RR_ref')
    plt.show()
    

