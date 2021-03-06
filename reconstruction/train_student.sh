BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='student'
ACC_FACTOR='8x'
CONTRAST='t1'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'_'${CONTRAST}
DATASET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
USMASK_PATH=${BASE_PATH}'/KD-MRI-CONTRAST/us_masks/'${DATASET_TYPE}

echo python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --dataset-path ${DATASET_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE} --contrast_type ${CONTRAST}
python train_student.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --dataset-path ${DATASET_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE} --contrast_type ${CONTRAST}
