echo "-------------------Automating the DL run--------------------"
FILE_GEN_PATH='/media/acrophase/pose1/Kapil/MultiRespDL/DL_Model/data_file_generator.py'
TESTBENCH_PATH='/media/acrophase/pose1/Kapil/MultiRespDL/DL_Model/new_testbench.py'
Data_Path='/media/acrophase/pose1/Kapil/ppg_dalia_data'
Saved_Model_Path='/media/acrophase/pose1/charan/BR_Uncertainty/DAYI_BIAN/SAVED_MODELS'
Annotation_Path='/media/acrophase/pose1/Kapil/MultiRespDL/DL_Model/annotation.pkl'
echo "---------------------File Generation Begins ----------------"
############################# ECG BASE MODEL ###############################
echo "python ${FILE_GEN_PATH} --data_path "Data_Path""
python ${FILE_GEN_PATH} --data_path ${Data_Path}
############################# NEW TESTBENCH BEGINS ###############################
echo "python ${TESTBENCH_PATH} --save_model_path "Saved_Model_Path""
python ${TESTBENCH_PATH} --save_model_path ${Saved_Model_Path} --annot_path ${Annotation_Path}
echo "---------------------Model Training Complete ----------------"
