import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal
import pywt
from scipy import interpolate
from scipy import fft
import pickle
import pandas as pd
from data_extraction import extract_data 
from filters import *
from hrv_analysis.extract_features import _create_interpolation_time, _create_time_info

srate = 700
fbpB , fbpA = band_pass(0.1,0.9,8)
fkern_lp_B ,fkern_lp_A = cheby_lp(6,25,1)              
fkern_hp_B , fkern_hp_A = cheby_hp(4,30,0.1)   

flpB_ref,flpA_ref = scipy.signal.cheby2(5,30 , 0.7/(srate/2),btype='lowpass')
fhpB_ref,fhpA_ref = scipy.signal.cheby2(4,20 ,0.1/(srate/2),btype='highpass')

def edr_adr_extraction(acc , rpeaks , rpeak_amplitudes,reference_resp, interp_rate = 4):
    '''
    inputs -- acc - Accelerometer signal extracted from dictionary returned by PPG_Dalia_data_extraction()
              rpeaks - R peak indices obtained from dictionary returned by PPG_Dalia_data_extraction()
              rpeak_amplitudes - R peak amplitudes obtained from dictionary returned by PPG_Dalia_data_extraction()
              reference_resp --  reference respiratory signal.
    
    outputs -- Function returns edr signals by HRV and RPEAK amplitude variations and ADR signal from accelerometer
               Final reference respiratory signal.
    Description -- Function takes the ACC, RPEAKS, Rpeak amplitudes for a particular subject and then calculate
                  the respiratory signal based of HRV, rpeak amplitude variations and using adaptive filtering
                  for accelerometer data. 
    '''
    final_edr_hrv = []
    final_edr_rpeak = []
    final_adr = []
    final_ref_resp = []
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
        filt_sig = scipy.signal.filtfilt(fbpB , fbpA , interpolated_signal)
        #filt_sig = np.append(filt_sig , np.zeros(128 - len(filt_sig)))
        final_edr_hrv.append(filt_sig)
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
        filt_sig = scipy.signal.filtfilt(fbpB,fbpA , interpolated_signal_rp)
        #filt_sig = np.append(filt_sig , np.zeros(128 - len(filt_sig)))
        final_edr_rpeak.append(filt_sig)
        i+=1
    #-------------------------RESPIRATORY SIGNAL BY ACCELEROMETER-------------------------
    # calculate the fft of accelerometer data and then select the spectrum between
    # the frequency range of 0.1Hz - 1Hz the frequency correspond to the maximum
    # power will be taken as central frequency and then that will decide the 
    # lower cut off frequency or upper cuttoff frequency of the filter to obtain
    # the respiratory signal.
    j=0
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
        resample_sig = scipy.signal.resample(final_signal , len(final_edr_rpeak[j]))
        #resample_sig = np.append(resample_sig , np.zeros(128 - len(resample_sig)))
        final_adr.append(resample_sig)
        j+=1
    
    k = 0
    for item in reference_resp:
        lp_filt = scipy.signal.filtfilt(flpB_ref,flpA_ref , item)
        hp_filt = scipy.signal.filtfilt(fhpB_ref,fhpA_ref , lp_filt)
        resmp_signal = scipy.signal.resample(hp_filt , len(final_edr_rpeak[k]))
        #resmp_signal = np.append(resmp_signal , np.zeros(128 - len(resmp_signal)))
        final_ref_resp.append(resmp_signal)
        k+=1
    return final_edr_hrv ,final_edr_rpeak ,final_adr,final_ref_resp

