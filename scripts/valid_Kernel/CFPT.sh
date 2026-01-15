#!/bin/bash
MAX_JOBS=48
GPUS=(0 1 2 3 4 5 6 7)
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

DATA_ROOT=$USRDIR/dataset
EXP_NAME=debug
seed=2023
des='CFPT'

model_name=CFPT
auxi_mode=kernel_balancing
# datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Traffic PEMS03 PEMS08)
datasets=(M5)


# hyper-parameters
dst=ETTh1

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=8
lradj=type1
period=24
beta=0.5
d_model=512
rda=1
rdb=1
kernel_size=3
e_layers=3
dropout=0.0
time_feature_types="['HourOfDay']"
rerun=0


# NOTE: ETTh1 settings

train_epochs=100
patience=15
pl_list=(96 192 336 720)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)


for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.5 d_model=512 kernel_size=3;;
        192) beta=0.5 d_model=512 kernel_size=3;;
        336) beta=0.7 d_model=512 kernel_size=5;;
        720) beta=0.3 d_model=512 kernel_size=2;;
    esac
    # case $pl in
    #     96) lr=0.0001 batch_size=8 beta=0.5 d_model=512 kernel_size=3;;
    #     192) lr=0.0001 batch_size=16 beta=0.5 d_model=512 kernel_size=3;;
    #     336) lr=0.0001 batch_size=16 beta=0.7 d_model=512 kernel_size=5;;
    #     720) lr=0.0001 batch_size=16 beta=0.3 d_model=512 kernel_size=2;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=24
beta=0.6
d_model=256
rda=1
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['HourOfDay']"
rerun=0

train_epochs=100
patience=15
pl_list=(96 192 336 720)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: ETTh2 settings


for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.6 d_model=256 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay']";;
        192) beta=0.4 d_model=1024 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay']";;
        336) beta=0.9 d_model=512 kernel_size=3 e_layers=6 time_feature_types="['HourOfDay']";;
        720) beta=0.4 d_model=1024 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay','MonthOfYear','SeasonOfYear']";;
    esac
    # case $pl in
    #     96) lr=0.0001 batch_size=4 beta=0.6 d_model=256 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay']";;
    #     192) lr=0.0001 batch_size=16 beta=0.4 d_model=1024 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay']";;
    #     336) lr=0.0001 batch_size=128 beta=0.9 d_model=512 kernel_size=3 e_layers=6 time_feature_types="['HourOfDay']";;
    #     720) lr=0.0001 batch_size=32 beta=0.4 d_model=1024 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay','MonthOfYear','SeasonOfYear']";;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=24
beta=0.6
d_model=256
rda=1
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['MinuteOfHour','HourOfDay']"
rerun=0

train_epochs=100
patience=15
pl_list=(96 192 336 720)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: ETTm1 settings



for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.6 d_model=1024 kernel_size=2 e_layers=3;;
        192) beta=0.9 d_model=512 kernel_size=2 e_layers=1;;
        336) beta=0.9 d_model=256 kernel_size=2 e_layers=1;;
        720) beta=0.9 d_model=512 kernel_size=2 e_layers=1;;
    esac
    # case $pl in
    #     96) lr=0.0001 batch_size=16 beta=0.6 d_model=1024 kernel_size=2 e_layers=3;;
    #     192) lr=0.0001 batch_size=8 beta=0.9 d_model=512 kernel_size=2 e_layers=1;;
    #     336) lr=0.0001 batch_size=8 beta=0.9 d_model=256 kernel_size=2 e_layers=1;;
    #     720) lr=0.0001 batch_size=8 beta=0.9 d_model=512 kernel_size=2 e_layers=1;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=24
beta=0.6
d_model=256
rda=1
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['HourOfDay']"
rerun=0

train_epochs=100
patience=15
pl_list=(96 192 336 720)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: ETTm2 settings



for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
        192) beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
        336) beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
        720) beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
    esac
    # case $pl in
    #     96) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
    #     192) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
    #     336) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
    #     720) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=24
beta=0.6
d_model=256
rda=8
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['HourOfDay','DayOfWeek','SeasonOfYear']"
rerun=0

train_epochs=100
patience=15
pl_list=(96 192 336 720)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: ECL settings


for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek']";;
        192) beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek']";;
        336) beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek','SeasonOfYear']";;
        720) beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek','SeasonOfYear']";;
    esac
    # case $pl in
    #     96) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek']";;
    #     192) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek']";;
    #     336) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek','SeasonOfYear']";;
    #     720) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek','SeasonOfYear']";;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=24
beta=0.6
d_model=256
rda=4
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['HourOfDay','DayOfWeek']"
rerun=0

train_epochs=100
patience=15
pl_list=(96 192 336 720)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: Traffic settings


for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.3 d_model=512 kernel_size=5 e_layers=6;;
        192) beta=0.3 d_model=512 kernel_size=5 e_layers=1;;
        336) beta=0.3 d_model=512 kernel_size=5 e_layers=1;;
        720) beta=0.3 d_model=512 kernel_size=5 e_layers=3;;
    esac
    # case $pl in
    #     96) lr=0.01 batch_size=4 beta=0.3 d_model=512 kernel_size=5 e_layers=6;;
    #     192) lr=0.01 batch_size=4 beta=0.3 d_model=512 kernel_size=5 e_layers=1;;
    #     336) lr=0.01 batch_size=4 beta=0.3 d_model=512 kernel_size=5 e_layers=1;;
    #     720) lr=0.01 batch_size=16 beta=0.3 d_model=512 kernel_size=5 e_layers=3;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=24
beta=0.6
d_model=256
rda=4
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['HourOfDay','SeasonOfYear']"
rerun=0

train_epochs=100
patience=15
pl_list=(96 192 336 720)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: Weather settings


for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.6 d_model=512 kernel_size=5 e_layers=3;;
        192) beta=0.6 d_model=256 kernel_size=2 e_layers=3;;
        336) beta=0.9 d_model=256 kernel_size=5 e_layers=3;;
        720) beta=0.9 d_model=128 kernel_size=5 e_layers=3;;
    esac
    # case $pl in
    #     96) lr=0.005 batch_size=128 beta=0.6 d_model=512 kernel_size=5 e_layers=3;;
    #     192) lr=0.005 batch_size=128 beta=0.6 d_model=256 kernel_size=2 e_layers=3;;
    #     336) lr=0.005 batch_size=128 beta=0.9 d_model=256 kernel_size=5 e_layers=3;;
    #     720) lr=0.005 batch_size=128 beta=0.9 d_model=128 kernel_size=5 e_layers=3;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=6
beta=0.6
d_model=256
rda=8
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['HourOfDay']"
rerun=0

train_epochs=100
patience=15
pl_list=(12 24 36 48)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: PEMS03 settings


for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        24) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        36) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        48) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    esac
    # case $pl in
    #     12) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    #     24) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    #     36) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    #     48) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=4
lradj=type1
period=6
beta=0.6
d_model=256
rda=8
rdb=1
kernel_size=2
e_layers=1
dropout=0.0
time_feature_types="['HourOfDay']"
rerun=0

train_epochs=100
patience=15
pl_list=(12 24 36 48)
alpha_list=(0.2 0.4 0.6)
lr_list=(0.0002 0.0001)
inner_lr_list=(0.0005 0.0002 0.0001)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01 0.001)
J_list=(3)
inner_step_list=(1 3 5)
gamma_list=(1 2)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32 64)
# NOTE: PEMS08 settings



for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        24) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        36) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        48) beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    esac
    # case $pl in
    #     12) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    #     24) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    #     36) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    #     48) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast

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
done
done
done
done
done
done
done
done





# hyper-parameters
dst=M5

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=8
lradj=type1
period=4
beta=0.5
d_model=512
rda=1
rdb=1
kernel_size=3
e_layers=3
dropout=0.0
time_feature_types="['DayOfYear','DayOfMonth']"
rerun=1


# NOTE: M5 settings

train_epochs=100
patience=15
# pl_list=(8 12 20 28)
pl_list=(28)
alpha_list=(0.5)
lr_list=(0.0002)
inner_lr_list=(0.0005)
inner_optim_list=(adam)
normed_list=(1)
joint_forecast_list=(1)
auxi_type_list=(akb)
kernel_type_list=(exp)
auxi_loss_list=(AKB)
C_list=(0.01)
J_list=(3)
inner_step_list=(3)
gamma_list=(1)
reg_list=(0.0001)
solver_type_list=(exact)
lradj_list=(type1)
bs_list=(32)


for lr in ${lr_list[@]}; do
for batch_size in ${bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for alpha in ${alpha_list[@]}; do
for solver_type in ${solver_type_list[@]}; do
case ${solver_type} in
    exact) _inner_lr_list=(0.0005) _inner_optim_list=(adam) _inner_step_list=(1) _reg_list=(${reg_list[@]});;
    optim) _inner_lr_list=(${inner_lr_list[@]}) _inner_optim_list=(${inner_optim_list[@]}) _inner_step_list=(${inner_step_list[@]}) _reg_list=(0.001);;
esac
for inner_lr in ${_inner_lr_list[@]}; do
for inner_optim in ${_inner_optim_list[@]}; do
for inner_step in ${_inner_step_list[@]}; do
for reg in ${_reg_list[@]}; do
for J in ${J_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
case ${auxi_loss} in
    AKB) _C_list=(${C_list[@]});;
    *) _C_list=(0.0);;
esac
for C in ${_C_list[@]}; do
for gamma in ${gamma_list[@]}; do
for kernel_type in ${kernel_type_list[@]}; do
for normed in ${normed_list[@]}; do
for auxi_type in ${auxi_type_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) beta=0.5 d_model=512 kernel_size=3;;
        192) beta=0.5 d_model=512 kernel_size=3;;
        336) beta=0.7 d_model=512 kernel_size=5;;
        720) beta=0.3 d_model=512 kernel_size=2;;
    esac
    # case $pl in
    #     96) lr=0.0001 batch_size=8 beta=0.5 d_model=512 kernel_size=3;;
    #     192) lr=0.0001 batch_size=16 beta=0.5 d_model=512 kernel_size=3;;
    #     336) lr=0.0001 batch_size=16 beta=0.7 d_model=512 kernel_size=5;;
    #     720) lr=0.0001 batch_size=16 beta=0.3 d_model=512 kernel_size=2;;
    # esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --root_path $DATA_ROOT/m5/ \
            --data_path m5.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data m5 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 10 \
            --dec_in 10 \
            --c_out 10 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast \
            --output_vis \
            --inverse

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
done
done
done
done
done
done
done
done



wait