from ppg_dalia_data_extraction import extract_data
from edr_adr_signal_extraction import edr_adr_extraction
from rqi_extraction import rqi_extraction
from rr_extraction import rr_extraction
import scipy
import numpy as np
import argparse
from machine_learning import *
import matplotlib.pyplot as plt
import re
import pandas as pd
import os
import pickle as pkl
import datetime as dt
from plots import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import seaborn as sns
import time
import datetime as dt

# Path for the feature .pkl file.
# path_freq = 'C:/Users/ee19s/Desktop/HR/PPG_FieldStudy/FREQ_MORPH_FEATURES.pkl'

bool_variable = True
# final_rr = []
final_error = []
if __name__ == "__main__":
    # parse the data path where .pkl file is present.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data")
    parser.add_argument(
        "--input_features",
        default="freq_morph",
        type=str,
        help="'freq', 'morph' or 'freq_morph'",
    )
    args = parser.parse_args()

    retreive_features = False
    # Check if features are already retreived in .pkl file or not.
    for file_name in os.listdir(args.data_path):
        if file_name.endswith("_FEATURES.pkl"):
            retreive_features = True
            path_to_pkl_file = os.path.join(args.data_path, file_name)
            break

    # Features are not retreived the execute the steps below or go to else case.
    if not (retreive_features):
        # path = 'C:/Users/ee19s/Desktop/HR/PPG_FieldStudy'
        srate = 700
        window_len = 32 * srate
        interp_freq = 4
        ecg_resp_win_length = 32 * interp_freq
        lags_ecg = np.arange(4, 45)
        lags_acc = np.arange(7, 70)
        # call the ppg_dalia_data_extraction function and extract the different signals
        data = extract_data(args.data_path, srate, window_len)

        for item in enumerate(data.keys()):
            patient_id = item[1]
            rpeaks = data[patient_id]["ECG"]["RPEAKS"]
            amps = data[patient_id]["ECG"]["AMPLITUDES"]
            acc = data[patient_id]["ACC"]["ACC_DATA"]
            resp = data[patient_id]["RESP"]["RESP_DATA"]
            activity_id = data[patient_id]["ACTIVITY_ID"]
            # import pdb;pdb.set_trace()
            # Filtering only for respiration signal to calculate the reference breathing rate per minute.
            flpB, flpA = scipy.signal.cheby2(5, 30, 0.7 / (srate / 2), btype="lowpass")
            fhpB, fhpA = scipy.signal.cheby2(4, 20, 0.1 / (srate / 2), btype="highpass")
            final_resp = []
            for item_1 in resp:
                lp_filt = scipy.signal.filtfilt(flpB, flpA, item_1)
                final_resp.append(scipy.signal.filtfilt(fhpB, fhpA, lp_filt))
            # import pdb;pdb.set_trace()
            # Call the respiration signal extraction function on ecg and acc,
            # for ecg only rpeaks and peak amplitudes are required.
            edr_hrv, edr_rpeak, adr = edr_adr_extraction(acc, rpeaks, amps)
            # Call the rr_extraction function to extract average breath duration from different respiratory signals.
            (
                rr_hrv_duration,
                hrv_cov,
                hrv_mean_ptop,
                hrv_true_min_true_max,
                hrv_cov_min,
            ) = rr_extraction(edr_hrv, interp_freq)
            (
                rr_rpeak_duration,
                rpeak_cov,
                rpeak_mean_ptop,
                rpeak_true_min_true_max,
                rpeak_cov_min,
            ) = rr_extraction(edr_rpeak, interp_freq)
            (
                rr_adr_duration,
                adr_cov,
                adr_mean_ptop,
                adr_true_min_true_max,
                adr_cov_min,
            ) = rr_extraction(adr, (srate / 10))
            rr_resp_duration, _, _, _, _ = rr_extraction(final_resp, srate)

            # Calcualte the respiratory rate per minute using the fomula below.
            rr_hrv = 60 / (rr_hrv_duration / interp_freq)
            rr_rpeak = 60 / (rr_rpeak_duration / interp_freq)
            rr_adr = 60 / (rr_adr_duration / (srate / 10))
            rr_resp = 60 / (rr_resp_duration / srate)
            # Calculate the Absolute error.
            abs_error_hrv = np.abs(rr_hrv - rr_resp)
            abs_error_rpeak = np.abs(rr_rpeak - rr_resp)
            abs_error_adr = np.abs(rr_adr - rr_resp)
            # Call the RQI extraction function to calculate different values of RQI.
            hrv_fft, hrv_ar, hrv_ac, hrv_hjorth, hrv_fft_extra = rqi_extraction(
                edr_hrv, ecg_resp_win_length, interp_freq, lags_ecg
            )
            (
                rpeak_fft,
                rpeak_ar,
                rpeak_ac,
                rpeak_hjorth,
                rpeak_fft_extra,
            ) = rqi_extraction(edr_rpeak, ecg_resp_win_length, interp_freq, lags_ecg)
            adr_fft, adr_ar, adr_ac, adr_hjorth, adr_fft_extra = rqi_extraction(
                adr, window_len, srate, lags_acc
            )
            # Frame the frequency feature metrics.

            int_part = re.findall(r"\d+", patient_id)
            freq_features = np.hstack(
                (
                    hrv_fft.reshape(-1, 1),
                    hrv_ar.reshape(-1, 1),
                    hrv_ac.reshape(-1, 1),
                    hrv_hjorth.reshape(-1, 1),
                    hrv_fft_extra.reshape(-1, 1),
                    rpeak_fft.reshape(-1, 1),
                    rpeak_ar.reshape(-1, 1),
                    rpeak_ac.reshape(-1, 1),
                    rpeak_hjorth.reshape(-1, 1),
                    rpeak_fft_extra.reshape(-1, 1),
                    adr_fft.reshape(-1, 1),
                    adr_ar.reshape(-1, 1),
                    adr_ac.reshape(-1, 1),
                    adr_hjorth.reshape(-1, 1),
                    adr_fft_extra.reshape(-1, 1),
                    np.array(activity_id).reshape(-1, 1),
                    rr_hrv.reshape(-1, 1),
                    rr_rpeak.reshape(-1, 1),
                    rr_adr.reshape(-1, 1),
                    rr_resp.reshape(-1, 1),
                    abs_error_hrv.reshape(-1, 1),
                    abs_error_rpeak.reshape(-1, 1),
                    abs_error_adr.reshape(-1, 1),
                    np.array([int(int_part[0])] * len(hrv_fft)).reshape(-1, 1),
                )
            )
            # Frame the morphological feature metrics.
            morph_features = np.hstack(
                (
                    np.array(hrv_cov).reshape(-1, 1),
                    np.array(hrv_mean_ptop).reshape(-1, 1),
                    np.array(hrv_true_min_true_max).reshape(-1, 1),
                    np.array(hrv_cov_min).reshape(-1, 1),
                    np.array(rpeak_cov).reshape(-1, 1),
                    np.array(rpeak_mean_ptop).reshape(-1, 1),
                    np.array(rpeak_true_min_true_max).reshape(-1, 1),
                    np.array(rpeak_cov_min).reshape(-1, 1),
                    np.array(adr_cov).reshape(-1, 1),
                    np.array(adr_mean_ptop).reshape(-1, 1),
                    np.array(adr_true_min_true_max).reshape(-1, 1),
                    np.array(adr_cov_min).reshape(-1, 1),
                    np.array(activity_id).reshape(-1, 1),
                    rr_hrv.reshape(-1, 1),
                    rr_rpeak.reshape(-1, 1),
                    rr_adr.reshape(-1, 1),
                    rr_resp.reshape(-1, 1),
                    abs_error_hrv.reshape(-1, 1),
                    abs_error_rpeak.reshape(-1, 1),
                    abs_error_adr.reshape(-1, 1),
                    np.array([int(int_part[0])] * len(hrv_fft)).reshape(-1, 1),
                )
            )

            mixed_features = np.hstack(
                (
                    hrv_fft.reshape(-1, 1),
                    hrv_ar.reshape(-1, 1),
                    hrv_ac.reshape(-1, 1),
                    np.array(hrv_cov).reshape(-1, 1),
                    np.array(hrv_mean_ptop).reshape(-1, 1),
                    np.array(hrv_true_min_true_max).reshape(-1, 1),
                    np.array(hrv_cov_min).reshape(-1, 1),
                    hrv_hjorth.reshape(-1, 1),
                    hrv_fft_extra.reshape(-1, 1),
                    rpeak_fft.reshape(-1, 1),
                    rpeak_ar.reshape(-1, 1),
                    rpeak_ac.reshape(-1, 1),
                    np.array(rpeak_cov).reshape(-1, 1),
                    np.array(rpeak_mean_ptop).reshape(-1, 1),
                    np.array(rpeak_true_min_true_max).reshape(-1, 1),
                    np.array(rpeak_cov_min).reshape(-1, 1),
                    rpeak_hjorth.reshape(-1, 1),
                    rpeak_fft_extra.reshape(-1, 1),
                    adr_fft.reshape(-1, 1),
                    adr_ar.reshape(-1, 1),
                    adr_ac.reshape(-1, 1),
                    np.array(adr_cov).reshape(-1, 1),
                    np.array(adr_mean_ptop).reshape(-1, 1),
                    np.array(adr_true_min_true_max).reshape(-1, 1),
                    np.array(adr_cov_min).reshape(-1, 1),
                    adr_hjorth.reshape(-1, 1),
                    adr_fft_extra.reshape(-1, 1),
                    np.array(activity_id).reshape(-1, 1),
                    rr_hrv.reshape(-1, 1),
                    rr_rpeak.reshape(-1, 1),
                    rr_adr.reshape(-1, 1),
                    rr_resp.reshape(-1, 1),
                    abs_error_hrv.reshape(-1, 1),
                    abs_error_rpeak.reshape(-1, 1),
                    abs_error_adr.reshape(-1, 1),
                    np.array([int(int_part[0])] * len(hrv_fft)).reshape(-1, 1),
                )
            )
            # Stack the features in array.
            if item[0] == 0:
                freq_features_all_patients = freq_features
                morph_features_all_patients = morph_features
                mixed_features_all_patients = mixed_features
            else:
                freq_features_all_patients = np.vstack(
                    (freq_features_all_patients, freq_features)
                )
                morph_features_all_patients = np.vstack(
                    (morph_features_all_patients, morph_features)
                )
                mixed_features_all_patients = np.vstack(
                    (mixed_features_all_patients, mixed_features)
                )
        # Column names of morphological and frequency based features.
        col_names_morph = [
            "hrv_cov",
            "hrv_mean_ptop",
            "hrv_true_max_true_min",
            "hrv_cov_min",
            "rpeak_cov",
            "rpeak_mean_ptop",
            "rpeak_true_max_true_min",
            "rpeak_cov_min",
            "adr_cov",
            "adr_mean_ptop",
            "adr_true_max_true_min",
            "adr_cov_min",
            "activity_id",
            "rr_hrv",
            "rr_rpeak",
            "rr_adr",
            "rr_resp",
            "error_hrv",
            "error_rpeak",
            "error_adr",
            "patient_id",
        ]
        col_names_freq = [
            "rqi_fft_hrv",
            "rqi_ar_hrv",
            "rqi_ac_hrv",
            "rqi_hjorth_hrv",
            "rqi_fft_extra_hrv",
            "rqi_fft_rpeak",
            "rqi_ar_rpeak",
            "rqi_ac_rpeak",
            "rqi_hjorth_rpeak",
            "rqi_fft_extra_rpeak",
            "rqi_fft_adr",
            "rqi_ar_adr",
            "rqi_ac_adr",
            "rqi_hjorth_adr",
            "rqi_fft_extra_adr",
            "activity_id",
            "rr_hrv",
            "rr_rpeak",
            "rr_adr",
            "rr_resp",
            "error_hrv",
            "error_rpeak",
            "error_adr",
            "patient_id",
        ]
        col_names_mixed = [
            "rqi_fft_hrv",
            "rqi_ar_hrv",
            "rqi_ac_hrv",
            "hrv_cov",
            "hrv_mean_ptop",
            "hrv_true_max_true_min",
            "hrv_cov_min",
            "rqi_hjorth_hrv",
            "rqi_fft_extra_hrv",
            "rqi_fft_rpeak",
            "rqi_ar_rpeak",
            "rqi_ac_rpeak",
            "rpeak_cov",
            "rpeak_mean_ptop",
            "rpeak_true_max_true_min",
            "rpeak_cov_min",
            "rqi_hjorth_rpeak",
            "rqi_fft_extra_rpeak",
            "rqi_fft_adr",
            "rqi_ar_adr",
            "rqi_ac_adr",
            "adr_cov",
            "adr_mean_ptop",
            "adr_true_max_true_min",
            "adr_covn_min",
            "rqi_hjorth_adr",
            "rqi_fft_extra_adr",
            "activity_id",
            "rr_hrv",
            "rr_rpeak",
            "rr_adr",
            "rr_resp",
            "error_hrv",
            "error_rpeak",
            "error_adr",
            "patient_id",
        ]
        # Store the features in pandas dataframe.
        df_freq = pd.DataFrame(freq_features_all_patients, columns=col_names_freq)
        df_morph = pd.DataFrame(morph_features_all_patients, columns=col_names_morph)
        df_mixed = pd.DataFrame(mixed_features_all_patients, columns=col_names_mixed)
        df_freq["patient_id"] = df_freq["patient_id"].astype(int)
        df_morph["patient_id"] = df_morph["patient_id"].astype(int)
        df_mixed["patient_id"] = df_mixed["patient_id"].astype(int)
        df_freq.to_pickle("FREQ_FEATURES.pkl")
        df_morph.to_pickle("MORPH_FEATURES.pkl")
        df_mixed.to_pickle("FREQ_MORPH_FEATURES.pkl")

    else:
        print(".............Pickle file containing rqi features exists.............")

        saved_model = dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
        results_path = os.path.join(args.data_path, "results")
        if not (os.path.isdir(results_path)):
            os.mkdir(results_path)
        current_model_path = os.path.join(results_path, saved_model)
        os.mkdir(current_model_path)

        print("............Created path.............")
        # Read the pickle files containing the features.
        data = pd.read_pickle(path_to_pkl_file)
        print("............Data read succesfully.............")
        # form the feature metrics such that it can be used in ml model
        # when given for 'freq' , 'morph', and both 'freq'+'morph'
        # based features.
        x_hrv_freq = data[["rqi_fft_hrv", "rqi_ar_hrv", "rqi_ac_hrv"]]
        x_rpeak_freq = data[["rqi_fft_rpeak", "rqi_ar_rpeak", "rqi_ac_rpeak"]]

        x_adr_freq = data[["rqi_fft_adr", "rqi_ar_adr", "rqi_ac_adr"]]
        x_hrv_morph = data[
            ["hrv_cov", "hrv_mean_ptop", "hrv_true_max_true_min", "hrv_cov_min"]
        ]
        x_rpeak_morph = data[
            ["rpeak_cov", "rpeak_mean_ptop", "rpeak_true_max_true_min", "rpeak_cov_min"]
        ]
        x_adr_morph = data[
            ["adr_cov", "adr_mean_ptop", "adr_true_max_true_min", "adr_covn_min"]
        ]

        activity_id = data.loc[:, "activity_id"]
        y_data = data.loc[:, "error_hrv":"patient_id"]
        # features metrics containing the relevent features to be fed into ml function. Its type is pd dataframe.
        feature_metrics = pd.concat(
            [
                x_hrv_freq,
                x_rpeak_freq,
                x_adr_freq,
                x_hrv_morph,
                x_rpeak_morph,
                x_adr_morph,
                activity_id,
                y_data,
            ],
            axis=1,
        )
        # Index of different respiratory rate values.
        index_hrv = data.columns.get_loc("rr_hrv")
        index_rpeak = data.columns.get_loc("rr_rpeak")
        index_adr = data.columns.get_loc("rr_adr")
        index_resp = data.columns.get_loc("rr_resp")
        if bool_variable == True:
            data = np.array(data)
            split = (data[:, -1]) >= 13
            # eximport pdb;pdb.set_trace()
            rr_hrv = data[split, index_hrv]
            rr_rpeak = data[split, index_rpeak]
            rr_adr = data[split, index_adr]
            rr_resp = data[split, index_resp]
            error_hrv = np.abs(rr_hrv - rr_resp).reshape(-1, 1)
            error_rpeak = np.abs(rr_rpeak - rr_resp).reshape(-1, 1)
            error_adr = np.abs(rr_adr - rr_resp).reshape(-1, 1)

            rmse_hrv = round(np.sqrt(np.mean(error_hrv ** 2)), 3)
            rmse_rpeak = round(np.sqrt(np.mean(error_rpeak ** 2)), 3)
            rmse_adr = round(np.sqrt(np.mean(error_adr ** 2)), 3)

            mae_hrv = round(np.mean(error_hrv), 4)
            mae_rpeak = round(np.mean(error_rpeak), 4)
            mae_adr = round(np.mean(error_adr), 4)
            activity_id = data[split, 27]
            # import pdb;pdb.set_trace()
            # Reshape to get two dimension.
            rr_hrv = rr_hrv.reshape(len(rr_hrv), -1)
            rr_rpeak = rr_rpeak.reshape(len(rr_rpeak), -1)
            rr_adr = rr_adr.reshape(len(rr_adr), -1)
            rr_resp = rr_resp.reshape(len(rr_resp), -1)
            # Create the machine learning object.

            ml = machine_learning(
                feature_metrics,
                args.input_features,
                is_patient_wise_split=bool_variable,
                is_save=True,
                model_save_path=current_model_path,
            )

            # List containing objects related to different models.
            print("....................Start of modelling...................")
            objects_list = [
                ml.ridge_regression(),
                ml.randomforest(),
                ml.supportvector(),
                ml.lasso_regression(),
                ml.bayesian_ridge(),
            ]
            print("....................End of modelling...................")

            # call the models and get the predicted values
            model_error_dict_rmse = {
                "Ridge": [],
                "RF": [],
                "SVM": [],
                "Lasso": [],
                "bRidge": [],
            }
            model_error_dict_mae = {
                "Ridge": [],
                "RF": [],
                "SVM": [],
                "Lasso": [],
                "bRidge": [],
            }
            model_dict_keys = ["Ridge", "RF", "SVM", "Lasso", "bRidge"]

            fusion_rr = []
            fusion_error_rmse = []
            fusion_error_mae = []
            time_list = []
            model_index = 1
            for index, item in tqdm(enumerate(objects_list)):
                # import pdb;pdb.set_trace()
                # start = time.time()
                hrv_pred, rpeak_pred, adr_pred = item
                # end = time.time()
                # time_list.append(end-start)
                if hrv_pred.ndim != 2 or rpeak_pred.ndim != 2 or adr_pred.ndim != 2:
                    hrv_pred = hrv_pred.reshape(len(hrv_pred), -1)
                    rpeak_pred = rpeak_pred.reshape(len(rpeak_pred), -1)
                    adr_pred = adr_pred.reshape(len(adr_pred), -1)

                # sum of all predicted values.
                # normal_sum = hrv_pred + rpeak_pred + adr_pred
                # normalise the weights,
                hrv_weights = np.exp((-1 * hrv_pred) / 5)  # 1 - (hrv_pred/normal_sum)
                rpeak_weights = np.exp(
                    (-1 * rpeak_pred) / 5
                )  # 1-(rpeak_pred/normal_sum)
                adr_weights = np.exp((-1 * adr_pred) / 5)  # 1 - (adr_pred/normal_sum)
                # obtain the net rr and error and store it in list.
                net_rr = (
                    rr_hrv * hrv_weights
                    + rr_rpeak * rpeak_weights
                    + rr_adr * adr_weights
                ) / (hrv_weights + rpeak_weights + adr_weights)
                error_rmse = np.sqrt(np.mean((net_rr - rr_resp) ** 2))
                error_mae = np.mean(np.abs(net_rr - rr_resp))
                fusion_rr.append(net_rr)
                # Append RMSE
                fusion_error_rmse.append(round(error_rmse, 4))
                # Append MAE
                fusion_error_mae.append(round(error_mae, 4))
                # Append RMSE in dictionary.
                model_error_dict_rmse[model_dict_keys[index]].append(rmse_hrv)
                model_error_dict_rmse[model_dict_keys[index]].append(rmse_rpeak)
                model_error_dict_rmse[model_dict_keys[index]].append(rmse_adr)
                model_error_dict_rmse[model_dict_keys[index]].append(
                    round(error_rmse, 4)
                )
                # Append MAE in dictionary.
                model_error_dict_mae[model_dict_keys[index]].append(mae_hrv)
                model_error_dict_mae[model_dict_keys[index]].append(mae_rpeak)
                model_error_dict_mae[model_dict_keys[index]].append(mae_adr)
                model_error_dict_mae[model_dict_keys[index]].append(round(error_mae, 4))
                error = np.abs(net_rr - rr_resp)
                # final_rr.append(net_rr)
                final_error.append(error)
            # import pdb;pdb.set_trace()
            # Store both rmse and mae in .csv file.
            with open(os.path.join(current_model_path, "results.txt"), "w") as f:
                f.write("----------------------- \n".format(args.input_features))
                f.write("Using {} features \n".format(args.input_features))
                f.write("----------------------- \n".format(args.input_features))
                for values, keys in enumerate(model_error_dict_rmse):
                    f.write(
                        "Model Type: {} - [RMSE HRV: {}, RMSE Rpeak: {}, RMSE ADR: {}, RMSE Fusion: {}] \n".format(
                            keys,
                            model_error_dict_rmse[keys][0],
                            model_error_dict_rmse[keys][1],
                            model_error_dict_rmse[keys][2],
                            model_error_dict_rmse[keys][3],
                        )
                    )
                    # f.write('Model Type: {} - [Error HRV: {}, Error Rpeak: {}, Error ADR: {}, Error Fusion: {}] \n'.format(keys, model_error_dict_mae[0], model_error_dict_mae[keys][1], model_error_dict_mae[keys][2], model_error_dict_mae[keys][3]))
                for value, keys in enumerate(model_error_dict_mae):
                    f.write(
                        "Model Type: {} - [MAE HRV: {}, MAE Rpeak: {}, MAE ADR: {}, MAE Fusion: {}] \n".format(
                            keys,
                            model_error_dict_mae[keys][0],
                            model_error_dict_mae[keys][1],
                            model_error_dict_mae[keys][2],
                            model_error_dict_mae[keys][3],
                        )
                    )
            # error_hrv = np.abs(rr_hrv - rr_resp)
            # error_rpeak = np.abs(rr_rpeak - rr_resp)
            # error_adr = np.abs(rr_adr - rr_resp)
# -----------------------------------------------FOR BOX PLOT---------------------------------------------
# Create the dataframe of absolute error according to the models.
model_index = 1
error_ml_model = np.hstack(
    (
        final_error[0].reshape(-1, 1),
        final_error[1].reshape(-1, 1),
        final_error[2].reshape(-1, 1),
        final_error[3].reshape(-1, 1),
        final_error[4].reshape(-1, 1),
    )
)
col_ml_model = ["Ridge", "Randomforest", "SVR", "Lasso", "Bayesian_Ridge"]
error_ml_model_df = pd.DataFrame(error_ml_model, columns=col_ml_model)

# Create the dataframe of absolute error according to the modality.
# To check the particular model change the model_index variable above in final_error[model_index].reshape(-1,1) between 0-4.
error_modality = np.hstack(
    (
        error_hrv.reshape(-1, 1),
        error_rpeak.reshape(-1, 1),
        error_adr.reshape(-1, 1),
        final_error[model_index].reshape(-1, 1),
    )
)
col_modality = ["error_hrv", "error_rpeak", "error_adr", "error_fused"]
error_modality_df = pd.DataFrame(error_modality, columns=col_modality)

# Frame the arrays according to the modalities and activities for the box plot.
array_hrv = np.concatenate(
    (
        error_hrv,
        np.array([0 for i in range(len(error_hrv))]).reshape(-1, 1),
        activity_id.reshape(-1, 1),
    ),
    axis=1,
)
array_rpeak = np.concatenate(
    (
        error_rpeak,
        np.array([1 for i in range(len(error_rpeak))]).reshape(-1, 1),
        activity_id.reshape(-1, 1),
    ),
    axis=1,
)
array_adr = np.concatenate(
    (
        error_adr,
        np.array([2 for i in range(len(error_adr))]).reshape(-1, 1),
        activity_id.reshape(-1, 1),
    ),
    axis=1,
)
array_fusion = np.concatenate(
    (
        final_error[model_index],
        np.array([3 for i in range(len(final_error[model_index]))]).reshape(-1, 1),
        activity_id.reshape(-1, 1),
    ),
    axis=1,
)

final_array = np.concatenate((array_hrv, array_rpeak, array_adr, array_fusion), axis=0)
data_frame = pd.DataFrame(final_array, columns=["error", "modality", "Activity_id"])
data_frame["modality"] = data_frame["modality"].astype("category")
data_frame["Activity_id"] = data_frame["Activity_id"].astype("category")
data_frame["modality"] = data_frame["modality"].cat.rename_categories(
    ["rr_int", "rpeak", "adr", "fused"]
)
data_frame["Activity_id"] = data_frame["Activity_id"].cat.rename_categories(
    [
        "Baseline",
        "Clean_baseline",
        "Cycling",
        "Driving",
        "Lunch",
        "No_activity",
        "Soccer",
        "Stairs",
        "Walking",
        "Working",
    ]
)

# Dataframe according to activity.
error_baseline = final_error[model_index][np.where(activity_id == 0)]
error_stairs = final_error[model_index][np.where(activity_id == 7)]
error_soccer = final_error[model_index][np.where(activity_id == 6)]
error_cycling = final_error[model_index][np.where(activity_id == 2)]
error_driving = final_error[model_index][np.where(activity_id == 3)]
error_lunch = final_error[model_index][np.where(activity_id == 4)]
error_walking = final_error[model_index][np.where(activity_id == 8)]
error_working = final_error[model_index][np.where(activity_id == 9)]

error_baseline_hrv = error_hrv[np.where(activity_id == 0)]
error_stairs_hrv = error_hrv[np.where(activity_id == 7)]
error_soccer_hrv = error_hrv[np.where(activity_id == 6)]
error_cycling_hrv = error_hrv[np.where(activity_id == 2)]
error_driving_hrv = error_hrv[np.where(activity_id == 3)]
error_lunch_hrv = error_hrv[np.where(activity_id == 4)]
error_walking_hrv = error_hrv[np.where(activity_id == 8)]
error_working_hrv = error_hrv[np.where(activity_id == 9)]

error_baseline_rpeak = error_rpeak[np.where(activity_id == 0)]
error_stairs_rpeak = error_rpeak[np.where(activity_id == 7)]
error_soccer_rpeak = error_rpeak[np.where(activity_id == 6)]
error_cycling_rpeak = error_rpeak[np.where(activity_id == 2)]
error_driving_rpeak = error_rpeak[np.where(activity_id == 3)]
error_lunch_rpeak = error_rpeak[np.where(activity_id == 4)]
error_walking_rpeak = error_rpeak[np.where(activity_id == 8)]
error_working_rpeak = error_rpeak[np.where(activity_id == 9)]

error_baseline_adr = error_adr[np.where(activity_id == 0)]
error_stairs_adr = error_adr[np.where(activity_id == 7)]
error_soccer_adr = error_adr[np.where(activity_id == 6)]
error_cycling_adr = error_adr[np.where(activity_id == 2)]
error_driving_adr = error_adr[np.where(activity_id == 3)]
error_lunch_adr = error_adr[np.where(activity_id == 4)]
error_walking_adr = error_adr[np.where(activity_id == 8)]
error_working_adr = error_adr[np.where(activity_id == 9)]

boxplot = sns.boxplot(data=error_modality_df, showfliers=False, width=0.5)
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=0)
plt.ylabel("Absolute Error")
plt.show()

ax = sns.boxplot(
    x="Activity_id", y="error", hue="modality", data=data_frame, showfliers=False
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.show()
# -----------------------------------------BLAND ALTMAN PLOT-------------------------------------
# Call the bland altman function.
bland_altman_plot(fusion_rr[model_index], rr_resp)

# ----------------------------------------PRINT THE ERRORS ACCORDING TO ACTIVITIES---------------
print("stairs mean abs error hrv {}".format(np.mean(np.abs(error_stairs_hrv))))
print("stairs mean abs error rpeak {}".format(np.mean(np.abs(error_stairs_rpeak))))
print("stairs mean abs error adr {}".format(np.mean(np.abs(error_stairs_adr))))
print("stairs mean abs error fusion {}".format(np.mean(np.abs(error_stairs))))
print("-----------------------------------------------------------")
print("cycling mean abs error hrv {}".format(np.mean(np.abs(error_cycling_hrv))))
print("cycling mean abs error rpeak {}".format(np.mean(np.abs(error_cycling_rpeak))))
print("cycling mean abs error adr {}".format(np.mean(np.abs(error_cycling_adr))))
print("cycling mean abs error fusion {}".format(np.mean(np.abs(error_cycling))))
print("-----------------------------------------------------------")
print("walking mean abs error hrv {}".format(np.mean(np.abs(error_walking_hrv))))
print("walking mean abs error rpeak {}".format(np.mean(np.abs(error_walking_rpeak))))
print("walking mean abs error adr {}".format(np.mean(np.abs(error_walking_adr))))
print("walking mean abs error fusion {}".format(np.mean(np.abs(error_walking))))

print("baseline mean abs error hrv {}".format(np.mean(np.abs(error_baseline_hrv))))
print("baseline mean abs error rpeak {}".format(np.mean(np.abs(error_baseline_rpeak))))
print("baseline mean abs error adr {}".format(np.mean(np.abs(error_baseline_adr))))
print("baseline mean abs error fusion {}".format(np.mean(np.abs(error_baseline))))

# ---------------------------------------------NOT OF IMMIDIATE USE---------------------------
"""
#        else:
#            # Split the data according to different modalities.
#            rr_hrv = data_freq['rr_hrv']
#            rr_rpeak = data_freq['rr_rpeak']
#            rr_adr = data_freq['rr_adr']
#            rr_resp = data_freq['rr_resp']
#            # Frame the data into array.
#            data_freq = np.array(data_freq)
#            rr_hrv = np.array(rr_hrv)
#            rr_rpeak = np.array(rr_rpeak)
#            rr_adr = np.array(rr_adr)
#            rr_resp = np.array(rr_resp)
            # Reshape the data to get 2 dimension.
#            rr_hrv = rr_hrv.reshape(len(rr_hrv) , -1)
#            rr_rpeak = rr_rpeak.reshape(len(rr_rpeak) , -1)
#            rr_adr = rr_adr.reshape(len(rr_adr) , -1)
#            rr_resp = rr_resp.reshape(len(rr_resp) , -1)
            # Create the machine learning objects.
#            ml = machine_learning(data_freq , is_patient_wise_split= bool_variable)
            # List containing objects related to different models.
#            objects_list = [ml.linear_regression() , ml.randomforest(), ml.supportvector(), ml.lasso_regression(),ml.bayesian_ridge()]
#            for item in objects_list:
#                hrv_pred , rpeak_pred , adr_pred,error_hrv , error_rpeak,error_adr = item
                
#                if hrv_pred.ndim != 2 or rpeak_pred.ndim!= 2 or adr_pred.ndim!= 2:
#                    hrv_pred = hrv_pred.reshape(len(hrv_pred) , -1)
#                    rpeak_pred = rpeak_pred.reshape(len(rpeak_pred) , -1)
#                    adr_pred = adr_pred.reshape(len(adr_pred) , -1)
                # sum of all predicted values.
#                normal_sum = hrv_pred + rpeak_pred + adr_pred
#                # normalise the weights,
#                hrv_weights = 1- (hrv_pred/normal_sum)
#                rpeak_weights = 1-(rpeak_pred/normal_sum)
#                adr_weights = 1 - (adr_pred/normal_sum)
                # get the respiratory rate corresponding to test data.
#                len_param = len(rr_hrv) - len(hrv_pred)
#                rr_hrv = rr_hrv[len_param:]
#                rr_rpeak = rr_rpeak[len_param:]
#                rr_adr = rr_adr[len_param:]
#                rr_resp = rr_resp[len_param :]
                # obtain the net rr and error and store it in list.
#                net_rr = (rr_hrv*hrv_weights + rr_rpeak*rpeak_weights + rr_adr*adr_weights)/(hrv_weights + rpeak_weights + adr_weights)
#                error = np.abs(net_rr - rr_resp)
#                final_rr.append(net_rr)
#                final_error.append(error)
#"""
