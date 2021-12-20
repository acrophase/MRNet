echo "------------------Automating the DL run--------------------"
FILE_GEN_PATH='/media/acrophase/pose1/Kapil/MultiRespDL/Dayi_Bian/data_file_generator.py'
TESTBENCH_PATH='/media/acrophase/pose1/Kapil/MultiRespDL/Dayi_Bian/new_testbench.py'
Data_Path='/media/acrophase/pose1/Kapil/ppg_dalia_data'
Saved_Model_Path='/media/acrophase/pose1/Kapil/MultiRespDL/Dayi_Bian/SAVED_MODELS'
Annotation_Path='/media/acrophase/pose1/Kapil/MultiRespDL/Dayi_Bian/annotation.pkl'
echo "---------------------File Generation Begins ----------------"
############################# DATA GENERATOR ###############################
echo "python ${FILE_GEN_PATH} --data_path "Data_Path""
python ${FILE_GEN_PATH} --data_path ${Data_Path}
############################# NEW TESTBENCH BEGINS ###############################
echo "python ${TESTBENCH_PATH} --save_model_path "Saved_Model_Path""
python ${TESTBENCH_PATH} --save_model_path ${Saved_Model_Path} --annot_path ${Annotation_Path}
echo "---------------------Model Training Complete ----------------"
