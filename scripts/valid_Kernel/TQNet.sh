#!/bin/bash
MAX_JOBS=2
GPUS=(0 1 2 3)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=./dataset
EXP_NAME=finetune
seed=2023
des='TQNet'

model_name=TQNet
auxi_mode=fft_ot
# datasets=(ECL Traffic Weather PEMS03 PEMS08)
datasets=(ETTh1)


# hyper-parameters
dst=ETTh1

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.5
cycle=24
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: ETTh1 settings



for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done





# hyper-parameters
dst=ETTh2

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.5
cycle=24
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: ETTh2 settings


for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done







# hyper-parameters
dst=ETTm1

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.5
cycle=96
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: ETTm1 settings



for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done








# hyper-parameters
dst=ETTm2

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.5
cycle=96
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: ETTm2 settings



for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=ECL

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.0
cycle=168
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(16)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: ECL settings


for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done







# hyper-parameters
dst=Traffic

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.0
cycle=168
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(16)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: Traffic settings


for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=Weather

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.5
cycle=144
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: Weather settings


for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=PEMS03

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=0
model_type=linear
dropout=0.0
cycle=288
rerun=0

pl_list=(12 24 36 48)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: PEMS03 settings


for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done








# hyper-parameters
dst=PEMS08

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=30
patience=5
test_batch_size=1
mask_factor=0.0
use_revin=1
model_type=linear
dropout=0.0
cycle=288
rerun=0

pl_list=(12 24 36 48)
alpha_list=(0.01 0.005)
lr_list=(0.001 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.0 0.5 1.0)
# NOTE: PEMS08 settings



for lr in ${lr_list[@]}; do
for reg_sk in ${reg_sk_list[@]}; do
for batch_size in ${bs_list[@]}; do
for eps in ${eps_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for var_weight in ${vw_list[@]}; do
for distance in ${distance_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}_${cycle}_${dropout}_${model_type}_${use_revin}
    OUTPUT_DIR="./results_OT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done
done






wait