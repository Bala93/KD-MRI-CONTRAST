BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='student'
CONTRAST_TYPE='t1'

ACC_FACTOR='8x'

TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${DATASET_TYPE}'_'${CONTRAST_TYPE}'/cartesian/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/results/'${CONTRAST_TYPE}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/results/'${CONTRAST_TYPE}

echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
