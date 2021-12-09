import pickle
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def extract_data (path, srate, window_length):
    '''
    Inputs --  path --  Path of the Data
               srate -- Sampling Rate
               Window_length -- Length of one window 32*srate in current case.
    Outputs -- Dictionary containing the infomation related to ECG, ACC, RESP signal.
    Description -- Function returns a dictionary which contains the ECG, ACC, RESP of every subject in
                PPG dalia dataset. Under dictionary with ECG as a key data contains ECG data in 32*srate 
               number of samples in one window it contain rpeaks and rpeak amplitude and data 
               according to different activities. Under ACC and RESP as key  ACC data in 32*srate
               segments and contains the data according to different activities.
    '''
    subjects = [i for i in sorted(os.listdir(path)) if not(i.endswith('pdf'))] 
    seconds_per_window = window_length / srate      
    data = {}
    
    for sub_id in tqdm(subjects):
        print('Subject Id is', sub_id) 
        windowed_ecg = []
        windowed_resp = []
        windowed_acc = []
        acc_x = []
        acc_y = []
        acc_z = []
        windowed_acc_y = []
        windowed_acc_z = []
        subpath = os.path.join(path , sub_id , sub_id+ ".pkl")
        subpath_activity = pd.read_csv(os.path.join(path , sub_id , sub_id+ "_activity.csv")) 
        subpath_activity = subpath_activity.rename(columns = {'# SUBJECT_ID':'subject_id'}) 
        subpath_activity['subject_id'] = subpath_activity.iloc[:,0].astype('category') 
        subpath_activity['activity_id'] = subpath_activity.subject_id.cat.codes 
        start_time = subpath_activity.iloc[: , 1].values 
        ### Obtaining activity annotation as a list ### 
        for index in range(1,len(subpath_activity)):
            if index == 1:
                annotation_per_window = [subpath_activity.iloc[index-1,2] for i in range(int(round(subpath_activity.iloc[index,1] / seconds_per_window)))]
                prev = round(subpath_activity.iloc[index,1] / seconds_per_window) * seconds_per_window 
            else:
                annotation_per_window += [subpath_activity.iloc[index-1,2] for i in range(int(round((subpath_activity.iloc[index,1] - prev) / seconds_per_window)))]
                prev = round(subpath_activity.iloc[index,1] / seconds_per_window) * seconds_per_window 
        with open (subpath , 'rb') as f:
            data_dict = pickle.load(f , encoding='bytes')
        ECG = data_dict[b'signal'][b'chest'][b'ECG']
        RESP = data_dict[b'signal'][b'chest'][b'Resp']
        acc_data = data_dict[b'signal'][b'chest'][b'ACC']
        rpeaks = data_dict[b'rpeaks']
        for item in acc_data:
            acc_x.append(item[0])
            acc_y.append(item[1])
            acc_z.append(item[2])
        #acc_x_axis = np.array(acc_x)
        acc_y_axis = np.array(acc_y)
        acc_z_axis = np.array(acc_z)
        ACC = acc_y_axis + acc_z_axis
        ECG = ECG.flatten()
        RESP = RESP.flatten()
        len_parameter = int(np.round(len(ECG)/window_length))
        RPEAKS = [np.array([]) for i in range (len_parameter)]
        amplitudes = [np.array([]) for i in range (len_parameter)] 
        for i in range(len_parameter):
            windowed_ecg.append(ECG[i*window_length : (i+1)*window_length])
            windowed_resp.append(RESP[i*window_length : (i+1)*window_length])
            windowed_acc.append(ACC[i*window_length : (i+1)*window_length])
            windowed_acc_y.append(acc_y_axis[i*window_length : (i+1)*window_length])
            windowed_acc_z.append(acc_z_axis[i*window_length : (i+1)*window_length])
            for item in rpeaks:
                if item >= i*window_length and item < (i+1)*window_length:
                    sub_factor = i*window_length
                    item1 = ECG[item]
                    RPEAKS[i] = np.append(RPEAKS[i] , item - sub_factor)
                    amplitudes[i] = np.append(amplitudes[i] , item1)
        while len(annotation_per_window)!= len(windowed_ecg):
            annotation_per_window.append(1)
        baseline_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==0)[0])]
        stairs_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==7)[0])]
        soccer_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==6)[0])]
        cycling_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==2)[0])]
        driving_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==3)[0])]
        lunch_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==4)[0])]
        walking_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==8)[0])]
        working_ecg = [windowed_ecg[item] for item in list(np.where(np.array(annotation_per_window)==9)[0])]

        baseline_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==0)[0])]
        stairs_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==7)[0])]
        soccer_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==6)[0])]
        cycling_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==2)[0])]
        driving_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==3)[0])]
        lunch_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==4)[0])]
        walking_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==8)[0])]
        working_acc = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==9)[0])]

        baseline_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==0)[0])]
        stairs_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==7)[0])]
        soccer_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==6)[0])]
        cycling_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==2)[0])]
        driving_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==3)[0])]
        lunch_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==4)[0])]
        walking_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==8)[0])]
        working_resp = [windowed_acc[item] for item in list(np.where(np.array(annotation_per_window)==9)[0])]
        
        baseline_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==0)[0])]
        stairs_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==7)[0])]
        soccer_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==6)[0])]
        cycling_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==2)[0])]
        driving_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==3)[0])]
        lunch_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==4)[0])]
        walking_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==8)[0])]
        working_rpeaks = [RPEAKS[item] for item in list(np.where(np.array(annotation_per_window)==9)[0])]

        baseline_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==0)[0])]
        stairs_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==7)[0])]
        soccer_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==6)[0])]
        cycling_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==2)[0])]
        driving_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==3)[0])]
        lunch_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==4)[0])]
        walking_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==8)[0])]
        working_amplitudes = [amplitudes[item] for item in list(np.where(np.array(annotation_per_window)==9)[0])]

        data.update({sub_id : {'ECG' : {'ECG_DATA' : windowed_ecg , 'RPEAKS': RPEAKS , 'AMPLITUDES': amplitudes,
                     'BASELINE_ECG':baseline_ecg , 'STAIRS_ECG' : stairs_ecg, 'SOCCER_ECG' : soccer_ecg
                       , 'CYCLING_ECG' : cycling_ecg , 'DRIVING_ECG' :  driving_ecg ,
                        'LUNCH_ECG' : lunch_ecg , 'WALKING_ECG': walking_ecg ,'WORKING_ECG':working_ecg,
                         'BASELINE_RPEAKS':baseline_rpeaks , 'STAIRS_RPEAKS': stairs_rpeaks ,
                         'SOCCER_RPEAKS': soccer_rpeaks , 'CYCLING_RPEAKS':cycling_resp,'DRIVING_RPEAKS':driving_rpeaks,
                         'LUNCH_RPEAKS':lunch_rpeaks,'WALKING_RPEAKS':walking_rpeaks,'WORKING_RPEAKS':working_rpeaks,
                         'BASELINE_AMPS':baseline_amplitudes,'STAIRS_AMPS':stairs_amplitudes,'SOCCER_AMPS':soccer_amplitudes,
                         'CYCLING_AMPS':cycling_amplitudes,'DRIVING_AMPS':driving_amplitudes,'LUNCH_AMPS':lunch_amplitudes,
                         'WALKING_AMPS':walking_amplitudes,'WORKING_RPEAKS':working_amplitudes }
                         ,'ACC':{'ACC_DATA': windowed_acc,'ACC_y':windowed_acc_y,'ACC_z':windowed_acc_y,
                          'BASELINE_ACC' :baseline_acc ,'STAIRS_ACC': stairs_acc, 'SOCCER_ACC': soccer_acc
                         , 'CYCLING_ACC' : cycling_acc, 'DRIVING_ACC': driving_acc , 'LUNCH_ACC': lunch_acc , 'WALKING_ACC': walking_acc
                         , 'WORKING_ACC': working_acc}
                          ,  'RESP': {'RESP_DATA': windowed_resp, 'BASELINE_RESP' : baseline_resp, 'STAIRS_RESP':stairs_resp,
                          'SOCCER_RESP':soccer_resp , 'CYCLING_RESP': cycling_resp , 'DRIVING_RESP': driving_resp,
                          'LUNCH_RESP':lunch_resp , 'WALKING_RESP': walking_resp , 'WORKING_RESP':working_resp  }
                          ,'ACTIVITY_ID': annotation_per_window}})
         
    return data
    
