import pandas as pd
from data_extraction import *
from resp_signal_extraction import *
from rr_extration import *
from sklearn.preprocessing import MinMaxScaler
import re
import pickle as pkl
import matplotlib.pyplot as plt
import sys
from scipy import signal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to data")
parser.add_argument("--srate", type=int, help="sampling rate", default=700)
parser.add_argument("--win_len", type=int, help="win length in secs", default=32)
args = parser.parse_args()

srate = args.srate
win_length = args.win_len * args.srate


data = extract_data(args.data_path, srate, win_length)

for item in enumerate(data.keys()):
    raw_ecg = []
    acc_x_axis = []
    acc_y_axis = []
    acc_z_axis = []
    patient_id = item[1]
    ecg = data[patient_id]["ECG"]["ECG_DATA"]
    rpeaks = data[patient_id]["ECG"]["RPEAKS"]
    amps = data[patient_id]["ECG"]["AMPLITUDES"]
    acc = data[patient_id]["ACC"]["ACC_DATA"]
    acc_x = data[patient_id]["ACC"]["ACC_X"]
    acc_y = data[patient_id]["ACC"]["ACC_Y"]
    acc_z = data[patient_id]["ACC"]["ACC_Z"]
    resp = data[patient_id]["RESP"]["RESP_DATA"]
    activity_id = data[patient_id]["ACTIVITY_ID"]
    scaler = MinMaxScaler()

    edr_hrv, edr_rpeak, adr, ref_resp = edr_adr_extraction(acc, rpeaks, amps, resp)

    for i in range(len(edr_hrv)):
        edr_hrv[i] = np.append(edr_hrv[i], np.zeros(128 - len(edr_hrv[i])))
        edr_rpeak[i] = np.append(edr_rpeak[i], np.zeros(128 - len(edr_rpeak[i])))
        adr[i] = np.append(adr[i], np.zeros(128 - len(adr[i])))
        ref_resp[i] = np.append(ref_resp[i], np.zeros(128 - len(ref_resp[i])))
        raw_ecg.append(signal.resample(ecg[i], 2048))
        acc_x_axis.append(signal.resample(acc_x[i], 2048))
        acc_y_axis.append(signal.resample(acc_y[i], 2048))
        acc_z_axis.append(signal.resample(acc_z[i], 2048))

    ecg_arr = np.array(raw_ecg)
    acc_x_arr = np.array(acc_x_axis)
    acc_y_arr = np.array(acc_y_axis)
    acc_z_arr = np.array(acc_z_axis)

    ref_rr_duration, _ = extremas_extraction(ref_resp)
    ref_rr = (60 * 4) / ref_rr_duration

    acc_y_axis = np.expand_dims(np.asarray(acc_y_axis), axis=-1)
    acc_z_axis = np.expand_dims(np.asarray(acc_z_axis), axis=-1)
    raw_ecg = np.expand_dims(np.asarray(raw_ecg), axis=-1)

    edr_hrv, edr_rpeak, adr, ref_resp = (
        np.expand_dims(np.asarray(edr_hrv), axis=-1),
        np.expand_dims(np.asarray(edr_rpeak), axis=-1),
        np.expand_dims(np.asarray(adr), axis=-1),
        np.expand_dims(np.asarray(ref_resp), axis=-1),
    )

    edr_hrv = scaler.fit_transform(edr_hrv.reshape(len(edr_hrv), len(edr_hrv[0])))
    edr_rpeak = scaler.fit_transform(
        edr_rpeak.reshape(len(edr_rpeak), len(edr_rpeak[0]))
    )
    adr = scaler.fit_transform(adr.reshape(len(adr), len(adr[0])))
    ref_resp = scaler.fit_transform(ref_resp.reshape(len(ref_resp), len(ref_resp[0])))
    ecg_arr = scaler.fit_transform(ecg_arr.reshape(len(raw_ecg), len(raw_ecg[0])))
    acc_y_arr = scaler.fit_transform(
        acc_y_arr.reshape(len(acc_y_arr), len(acc_y_arr[0]))
    )
    acc_z_arr = scaler.fit_transform(
        acc_z_arr.reshape(len(acc_z_arr), len(acc_z_arr[0]))
    )

    windowed_inp = np.concatenate(
        (
            np.expand_dims(edr_hrv, 1),
            np.expand_dims(edr_rpeak, 1),
            np.expand_dims(adr, 1),
        ),
        axis=1,
    )
    windowed_raw_sig = np.concatenate(
        (
            np.expand_dims(ecg_arr, 1),
            np.expand_dims(acc_y_arr, 1),
            np.expand_dims(acc_z_arr, 1),
        ),
        axis=1,
    )
    int_part = re.findall(r"\d+", patient_id)

    sub_activity_ids = np.hstack(
        (
            ref_rr.reshape(-1, 1),
            np.array(activity_id).reshape(-1, 1),
            np.array([int(int_part[0])] * len(edr_hrv)).reshape(-1, 1),
        )
    )

    if item[0] == 0:
        final_windowed_inp = windowed_inp
        final_windowed_op = np.array(ref_resp)
        final_windowed_raw = windowed_raw_sig
        final_sub_activity_ids = sub_activity_ids
    else:
        final_windowed_inp = np.vstack((final_windowed_inp, windowed_inp))
        final_windowed_op = np.vstack((final_windowed_op, ref_resp))
        final_sub_activity_ids = np.vstack((final_sub_activity_ids, sub_activity_ids))
        final_windowed_raw = np.vstack((final_windowed_raw, windowed_raw_sig))

# if(extract_pickle == True):
activity_df = pd.DataFrame(
    final_sub_activity_ids, columns=["Reference_RR", "activity_id", "patient_id"]
)
activity_df.to_pickle("annotation.pkl")
with open("output", "wb") as f:
    pkl.dump(final_windowed_op, f)

with open("input", "wb") as f:
    pkl.dump(final_windowed_inp, f)

with open("raw_signal.pkl", "wb") as f:
    pkl.dump(final_windowed_raw, f)
