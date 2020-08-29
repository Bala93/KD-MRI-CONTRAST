BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation_meta'
DATASET_TYPE='mrbrain'
MODEL_TYPE='student'
ACC_FACTOR='8x'

BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
DATASET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
USMASK_PATH=${BASE_PATH}'/KD-MRI-CONTRAST/us_masks/'${DATASET_TYPE}

echo python train_student.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --dataset-path ${DATASET_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE}
python train_student.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --dataset-path ${DATASET_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE}
