import tensorflow as tf
import os
import numpy as np
import random
SEED = 0
#------------------------------------------------------------------------------------
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
#------------------------------------------------------------------------------------
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)
#-----------------------------------------------------------------------------------
import pandas as pd
from data_extraction import *
from resp_signal_extraction import *
from rr_extration import *
from sklearn.preprocessing import MinMaxScaler
import re
import pickle as pkl
from model import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import evidential_deep_learning as edl
import matplotlib.pyplot as plt
import datetime
import sys

srate = 700
win_length = 32*srate
num_epochs = 100
train_test_split_id = 13
#config = input("Enter the configuration :")
data_path = '/media/acrophase/pose1/charan/BR_Uncertainty/ppg_dalia_data'
data = extract_data(data_path , srate , win_length)
  
#saved_model_path = os.path.join( 

for item in enumerate(data.keys()):
    patient_id = item[1]  
    ecg = data[patient_id]['ECG']['ECG_DATA']
    rpeaks = data[patient_id]['ECG']['RPEAKS']
    amps = data[patient_id]['ECG']['AMPLITUDES']
    acc = data[patient_id]['ACC']['ACC_DATA']
    resp = data[patient_id]['RESP']['RESP_DATA']
    activity_id = data[patient_id]['ACTIVITY_ID']
    scaler = MinMaxScaler()

    edr_hrv , edr_rpeak , adr , ref_resp = edr_adr_extraction(acc, rpeaks , amps , resp)

    for i in range(len(edr_hrv)):
        edr_hrv[i] = np.append(edr_hrv[i] , np.zeros(128 - len(edr_hrv[i])))
        edr_rpeak[i] = np.append(edr_rpeak[i] , np.zeros(128 - len(edr_rpeak[i])))
        adr[i] = np.append(adr[i] , np.zeros(128 - len(adr[i])))
        ref_resp[i] = np.append(ref_resp[i] , np.zeros(128 - len(ref_resp[i])))
    ref_rr_duration, _ =  extremas_extraction(ref_resp)
    ref_rr = (60*4)/ref_rr_duration

    edr_hrv , edr_rpeak , adr , ref_resp = np.expand_dims(np.asarray(edr_hrv), axis = -1), np.expand_dims(np.asarray(edr_rpeak), axis = -1)\
                               , np.expand_dims(np.asarray(adr), axis =-1) , np.expand_dims(np.asarray(ref_resp), axis =-1)
    
    edr_hrv = scaler.fit_transform(edr_hrv.reshape(len(edr_hrv),len(edr_hrv[0])))
    edr_rpeak = scaler.fit_transform(edr_rpeak.reshape(len(edr_rpeak),len(edr_rpeak[0])))
    adr = scaler.fit_transform(adr.reshape(len(adr),len(adr[0])))
    ref_resp = scaler.fit_transform(ref_resp.reshape(len(ref_resp),len(ref_resp[0])))

    windowed_inp = np.concatenate((np.expand_dims(edr_hrv, 1), np.expand_dims(edr_rpeak, 1), np.expand_dims(adr, 1)), axis = 1)
    int_part  = re.findall(r'\d+', patient_id)

    sub_activity_ids = np.hstack((ref_rr.reshape(-1,1),np.array(activity_id).reshape(-1,1), np.array([int(int_part[0])]*len(edr_hrv)).reshape(-1,1)))
    
    if item[0] == 0:
        final_windowed_inp = windowed_inp
        final_windowed_op = np.array(ref_resp)
        final_sub_activity_ids = sub_activity_ids
    else:
        final_windowed_inp = np.vstack((final_windowed_inp , windowed_inp))
        final_windowed_op = np.vstack((final_windowed_op , ref_resp))
        final_sub_activity_ids = np.vstack((final_sub_activity_ids , sub_activity_ids))

with open('output','rb') as f:
    output_data = pkl.load(f)

with open('input','rb') as f:
    input_data = pkl.load(f)

with open('raw_signal_2.pkl','rb') as f:
    raw_data = pkl.load(f)

input_data = np.transpose(input_data, (0,2,1))
raw_data = np.transpose(raw_data, (0,2,1))

input_data = np.around(input_data , decimals = 4)
raw_data = np.around(raw_data , decimals = 4)
output_data = np.around(output_data , decimals = 4)

annotation = pd.read_pickle('/media/acrophase/pose1/charan/BR_Uncertainty/ATTENTION/annotation.pkl')
reference_rr = (annotation['Reference_RR'].values).reshape(-1,1)
reference_rr = np.around(reference_rr , decimals = 4)

tensor_input = tf.convert_to_tensor(input_data , dtype = 'float32')
tensor_output = tf.convert_to_tensor(output_data , dtype = 'float32')
tensor_ref_rr = tf.convert_to_tensor(reference_rr, dtype = 'float32')
tensor_raw_data = tf.convert_to_tensor(raw_data, dtype = 'float32')

training_ids = annotation['patient_id'] < train_test_split_id

x_train_data = tensor_input[tf.convert_to_tensor(training_ids.values)]
x_test_data = tensor_input[tf.convert_to_tensor(~(training_ids.values))]
x_train_ref_rr = tensor_ref_rr[tf.convert_to_tensor(training_ids.values)]
x_test_ref_rr = tensor_ref_rr[tf.convert_to_tensor(~(training_ids.values))]
x_train_raw_sig = tensor_raw_data[tf.convert_to_tensor(training_ids.values)]
x_test_raw_sig = tensor_raw_data[tf.convert_to_tensor(~(training_ids.values))]

y_train_data = tensor_output[tf.convert_to_tensor(training_ids.values)]
y_test_data = tensor_output[tf.convert_to_tensor(~(training_ids.values))]

model_input_shape = (2048,3)
lr = 1e-5
optimizer = Adam(learning_rate = lr) 
model  = CNN(model_input_shape)
loss_fn = Huber()
save_path = '/home/hticpose/MultiRespDL/DAYI_BIAN/SAVED_MODELS'
results_path = os.path.join(save_path , str(lr))
if not(os.path.isdir(results_path)):
        os.mkdir(results_path)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_raw_sig , x_train_ref_rr))
train_dataset = train_dataset.shuffle(len(x_train_raw_sig)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_raw_sig , x_test_ref_rr))
test_dataset = test_dataset.batch(128)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'evi/logs/gradient_tape/'+current_time + '/train'
test_log_dir = 'evi/logs/gradient_tape/' +current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)  

print("Starting the training for : {}".format(item))
for epoch in range(num_epochs):
    print("starting the epoch : {}".format(epoch + 1))
    train_loss_list = []
    for step, (x_batch_train_raw , x_batch_train_ref_rr) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            output = model(x_batch_train_raw , training = True)
            loss_value = loss_fn(x_batch_train_ref_rr , output)
            train_loss_list.append(loss_value)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
        train_loss(loss_value)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)

        if step%10 == 0:
            print('Epoch [%d/%d], lter [%d] Loss: %.4f'
                    %(epoch+1, num_epochs, step+1, loss_value))
    print("net loss -- {}".format(np.mean(np.array(train_loss_list))))
    test_loss_list = []
    best_loss = 100000

    for step , (x_batch_test_raw, x_batch_test_ref_rr) in enumerate(test_dataset):
        test_output = model(x_batch_test_raw)
        test_loss_val = loss_fn(x_batch_test_ref_rr , test_output)
        test_loss(test_loss_val)
        test_loss_list.append(test_loss_val)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
                #print(test_output)
    mean_loss = (sum(test_loss_list) / len(test_loss_list)) 
    if mean_loss < best_loss:
        best_loss = mean_loss
        model.save_weights(os.path.join(results_path, 'best_model_1'+'lr_'+str(lr)+'_'+str(num_epochs)+'.h5'))
    print("validation loss -- {}".format(mean_loss)) 
    train_loss.reset_states()
    test_loss.reset_states()



