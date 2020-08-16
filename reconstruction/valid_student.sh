BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='kd' #student,kd

ACC_FACTOR='8x'
BATCH_SIZE=1
DEVICE='cuda:1'

CONTRAST='flair'

DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'_'${CONTRAST}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'_'${CONTRAST}'/results'

#echo python valid_student.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE} --contrast_type ${CONTRAST}
python valid_student.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE} --contrast_type ${CONTRAST}

