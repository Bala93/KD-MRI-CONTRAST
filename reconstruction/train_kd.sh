BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='kd'
ACC_FACTOR='8x'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:1'

CONTRAST='flair'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'_'${CONTRAST}
DATASET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
USMASK_PATH=${BASE_PATH}'/KD-MRI-CONTRAST/us_masks/'${DATASET_TYPE}

TEACHER_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_teacher/best_model.pt'
STUDENT_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_feature_'${CONTRAST}'/best_model.pt'

echo python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --dataset-path ${DATASET_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT}
python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --dataset-path ${DATASET_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT} --contrast_type ${CONTRAST}

