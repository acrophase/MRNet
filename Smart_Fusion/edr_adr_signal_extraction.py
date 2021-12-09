import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal
import pywt
from scipy import interpolate
from scipy import fft
import pickle
import pandas as pd
from ppg_dalia_data_extraction import extract_data 
from filters import *
from hrv_analysis.extract_features import _create_interpolation_time, _create_time_info

srate = 700
fbpB , fbpA = band_pass(0.1,0.9,8)
fkern_lp_B ,fkern_lp_A = cheby_lp(6,25,1)              
fkern_hp_B , fkern_hp_A = cheby_hp(4,30,0.1)   

def edr_adr_extraction(acc , rpeaks , rpeak_amplitudes, interp_rate = 4 , ds_factor = 10):
    '''
    inputs -- acc - Accelerometer signal extracted from dictionary returned by PPG_Dalia_data_extraction()
              rpeaks - R peak indices obtained from dictionary returned by PPG_Dalia_data_extraction()
              rpeak_amplitudes - R peak amplitudes obtained from dictionary returned by PPG_Dalia_data_extraction()
              interpolation rate = 4Hz(fixed)
              ds_factor -- Downsampling factor.

    outputs -- Function returns edr signals by HRV and RPEAK amplitude variations and ADR signal from accelerometer.
    Description -- Function takes the ACC, RPEAKS, Rpeak amplitudes for a particular subject and then calculate
                  the respiratory signal based of HRV, rpeak amplitude variations and using adaptive filtering
                  for accelerometer data. 
    '''
    final_edr_hrv = []
    final_edr_rpeak = []
    final_adr = []
    #-----------------------RESPIRATORY SIGNAL BY HRV------------------------------------
    # interpolate the rr interval using cubic spline interpolation and filter between 
    # 0.1Hz - 1Hz to obtain final edr
    for item in rpeaks:
        #import pdb;pdb.set_trace()
        rr_interval = (np.diff(item)/srate)*1000
        index = np.where(rr_interval == 0)
        rr_interval = np.delete(rr_interval , index)
        rr_times = _create_time_info(list(rr_interval))
        funct = interpolate.interp1d(x=rr_times, y=list(rr_interval), kind='cubic')
        timestamps_interpolation = _create_interpolation_time(rr_times, 4)
        interpolated_signal = funct(timestamps_interpolation)
        #time_stamp_hrv = np.arange(0,len(rr_interval))
        #time_interp_hrv = np.arange(time_stamp_hrv[0] , time_stamp_hrv[-1] , 1/interp_rate)
        #interpolated_signal = scipy.interpolate.griddata(time_stamp_hrv , rr_interval , time_interp_hrv , method='cubic')
        interpolated_signal = (interpolated_signal - np.mean(interpolated_signal))/np.std(interpolated_signal)
        final_edr_hrv.append(scipy.signal.filtfilt(fbpB , fbpA , interpolated_signal))
    #---------------------RESPIRATORY SIGNAL BY RPEAKS-----------------------------------
    # interpolate the rpeak amplitudes using cubic spline interpolation and filter between 
    # 0.1Hz - 1Hz to obtain final edr
    i = 0
    for item in rpeak_amplitudes:
        rr_interval = (np.diff(rpeaks[i])/srate)*1000
        index = np.where(rr_interval == 0)
        rr_interval = np.delete(rr_interval , index)
        item = np.delete(item,index)
        rr_times = _create_time_info(list(rr_interval))
        funct = interpolate.interp1d(x=rr_times, y=list(item[1:]), kind='cubic')
        timestamps_interpolation = _create_interpolation_time(rr_times, 4)
        interpolated_signal_rp = funct(timestamps_interpolation)
        #time_stamp_rpeak = np.arange(0 , len(item))
        #time_interp_rpeak = np.arange(time_stamp_rpeak[0] , time_stamp_rpeak[-1] , 1/interp_rate)
        #interpolated_signal_rp = scipy.interpolate.griddata(time_stamp_rpeak , item , time_interp_rpeak ,method='cubic' )
        interpolated_signal_rp = (interpolated_signal_rp - np.mean(interpolated_signal_rp))/np.std(interpolated_signal_rp)
        final_edr_rpeak.append(scipy.signal.filtfilt(fbpB,fbpA , interpolated_signal_rp))
        i+=1
    #-------------------------RESPIRATORY SIGNAL BY ACCELEROMETER-------------------------
    # calculate the fft of accelerometer data and then select the spectrum between
    # the frequency range of 0.1Hz - 1Hz the frequency correspond to the maximum
    # power will be taken as central frequency and then that will decide the 
    # lower cut off frequency or upper cuttoff frequency of the filter to obtain
    # the respiratory signal.
    for item in acc:
        lp_filt_sig = scipy.signal.filtfilt(fkern_lp_B , fkern_lp_A , item)
        hp_filt_sig = scipy.signal.filtfilt(fkern_hp_B , fkern_hp_A , lp_filt_sig)
        spectrum = np.absolute(scipy.fft.fft(hp_filt_sig)**2)
        freq = scipy.fft.fftfreq(len(spectrum) , d= 1/srate)
        upper_index = int(len(item)/srate + 1)
        lower_index = int((0.1*len(item))/srate)
        rel_freq = freq[lower_index:upper_index]
        rel_spectrum = spectrum[lower_index:upper_index]
        max_freq = rel_freq[np.argmax(rel_spectrum)]
        lower_cut_freq = max(0.1 , max_freq-0.4)
        upper_cut_freq = max_freq + 0.4
        flpB ,flpA = scipy.signal.cheby2(5,30,upper_cut_freq/(srate/2) , btype='lowpass')
        fhpB , fhpA = scipy.signal.cheby2(4,30, lower_cut_freq/(srate/2) , btype='highpass')
        lp_filt_acc = scipy.signal.filtfilt(flpB, flpA , hp_filt_sig)
        final_signal = scipy.signal.filtfilt(fhpB , fhpA, lp_filt_acc)
        final_adr.append(scipy.signal.decimate(final_signal , ds_factor))
    return final_edr_hrv , final_edr_rpeak , final_adr
