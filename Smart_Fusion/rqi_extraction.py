import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal
import pywt
from scipy import interpolate
from scipy import fft
import pickle
import pandas as pd
from ppg_dalia_data_extraction import extract_data 
from edr_adr_signal_extraction import edr_adr_extraction
from spectrum import *
srate = 700
def rqi_extraction(raw_signal ,nfft , srate, lags):
    '''
    inputs -- raw_signal -- Respiratory Signal 
              nfft - number of fft points
                    = 32*4 when respiratory signal is from ECG.
                    = 32*70 when respiratory signal is from ACC.
              srate - sampling rate 
                    = 4Hz for ECG based EDR
                    = 700Hz for ACC based ADR
               lags - lags for which autocorrelation RQI needs to be 
                      calculated. 
    output -- RQI_FFT - RQI from FFT
              RQI_Autoregression -  RQI from Autoregression.
              RQI_Autocorrelation - RQI from Autocorrelation
              RQI_hjorth -  RQI from Hjorth parameter.
              RQI_FFT_Extra -  RQI from FFT when divided be entire spectrum.
    Description -- This function takes the raw respiratory signals and the array of lags
                   and then based of FFT, Hjorth parameter, Autocorrelation and Autoregression gives
                   RQI values for a particular segment.
    '''
    model_order = np.arange(1,31)
    RQI_FFT = np.array([])
    RQI_FFT_extra = np.array([])
    RQI_hjorth = np.array([])
    RQI_Autocorrelation = np.array([])
    RQI_Autoregression = np.array([])
    for item in raw_signal:
        corr = np.array([])
        aic = np.array([])
        coefficients = []
        #-----------------------RQI BY FFT------------------
        spectrum = np.abs(scipy.fftpack.fft(item , nfft)**2)
        freq = scipy.fftpack.fftfreq(len(spectrum) , d = 1/srate)
        upper_index = int(nfft/srate + 1)
        lower_index = int((0.1*nfft)/srate)
        rel_freq = freq[lower_index:upper_index]
        rel_spectrum = spectrum[lower_index:upper_index]
        max_val_index = np.argmax(rel_spectrum)
        if max_val_index == 0:
            mpa = rel_spectrum[max_val_index] + rel_spectrum[max_val_index + 1] + rel_spectrum[max_val_index +2]
        if max_val_index == len(rel_spectrum)-1:
            mpa = rel_spectrum[max_val_index] + rel_spectrum[max_val_index - 1] + rel_spectrum[max_val_index -2]
        else:
            mpa = rel_spectrum[max_val_index -1] + rel_spectrum[max_val_index] + rel_spectrum[max_val_index +1]
        tra = np.sum(rel_spectrum)
        RQI_FFT = np.append(RQI_FFT , (mpa/tra))
        RQI_FFT_extra = np.append(RQI_FFT_extra , mpa/np.sum(spectrum))
        #-----------------------RQI BY HJORTH-------------------
        data = (item - np.mean(item))/np.std(item)
        first_deriv = np.diff(data)
        second_deriv = np.diff(data , 2) 
        var_zero = np.mean(data**2)
        var_d1 = np.mean(first_deriv**2)
        var_d2 = np.mean(second_deriv**2)
        #activity = var_zero
        mobility = np.sqrt(var_d1 / var_zero)
        complexilty = np.sqrt(var_d2 / var_d1) / mobility
        RQI_hjorth = np.append(RQI_hjorth , complexilty)
        #-----------------------RQI BY AUTOCORRELATION------------
        mean_val = np.mean(item)
        var_val = np.var(item)
        xp = item - mean_val
        for l in lags:
            corr = np.append(corr , (np.sum(xp[l:] * xp[:-l])/len(item)/var_val))
        RQI_Autocorrelation = np.append(RQI_Autocorrelation , max(corr))
        #-----------------------RQI BY AUTOREGRESSION--------------------
        for m in model_order:
            ar,var,_ = aryule(item , m)
            aic = np.append(aic ,AIC(len(item),var,m))
            coefficients.append(ar)
        min_aic_index = np.argmin(aic)
        ar_relevent = coefficients[min_aic_index]
        psd = arma2psd(ar_relevent , NFFT = 481)
        freq = scipy.fftpack.fftfreq(len(psd) , d = 1/2)
        rel_freq = freq[24:241]
        rel_spectrum = spectrum[24:241]
        max_psd_index = np.argmax(rel_spectrum)
        if max_psd_index < 4:
            peak_area = 0
            for i in range(0,9):
                peak_area = peak_area + rel_spectrum[(max_psd_index + i)]
        elif max_psd_index >= len(rel_spectrum) - 4:
            peak_area = 0
            for i in range(0,9):
                peak_area = peak_area + rel_spectrum[(max_psd_index - i)]
        else:
            peak_area = 0
            for i in range(-4,5):
                peak_area = peak_area + rel_spectrum[(max_psd_index + i)]            
        total_peak_area = np.sum(rel_spectrum)
        RQI_Autoregression = np.append(RQI_Autoregression ,(peak_area/total_peak_area))
    return RQI_FFT , RQI_Autoregression ,RQI_Autocorrelation , RQI_hjorth , RQI_FFT_extra


