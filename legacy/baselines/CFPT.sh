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
auxi_mode=rfft
datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Traffic PEMS03 PEMS08)


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

pl_list=(96 192 336 720)
# NOTE: ETTh1 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 batch_size=8 beta=0.5 d_model=512 kernel_size=3;;
        192) lr=0.0001 batch_size=16 beta=0.5 d_model=512 kernel_size=3;;
        336) lr=0.0001 batch_size=16 beta=0.7 d_model=512 kernel_size=5;;
        720) lr=0.0001 batch_size=16 beta=0.3 d_model=512 kernel_size=2;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(96 192 336 720)
# NOTE: ETTh2 settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 batch_size=4 beta=0.6 d_model=256 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay']";;
        192) lr=0.0001 batch_size=16 beta=0.4 d_model=1024 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay']";;
        336) lr=0.0001 batch_size=128 beta=0.9 d_model=512 kernel_size=3 e_layers=6 time_feature_types="['HourOfDay']";;
        720) lr=0.0001 batch_size=32 beta=0.4 d_model=1024 kernel_size=2 e_layers=1 time_feature_types="['HourOfDay','MonthOfYear','SeasonOfYear']";;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(96 192 336 720)
# NOTE: ETTm1 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 batch_size=16 beta=0.6 d_model=1024 kernel_size=2 e_layers=3;;
        192) lr=0.0001 batch_size=8 beta=0.9 d_model=512 kernel_size=2 e_layers=1;;
        336) lr=0.0001 batch_size=8 beta=0.9 d_model=256 kernel_size=2 e_layers=1;;
        720) lr=0.0001 batch_size=8 beta=0.9 d_model=512 kernel_size=2 e_layers=1;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(96 192 336 720)
# NOTE: ETTm2 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
        192) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
        336) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
        720) lr=0.0001 batch_size=4 beta=0.3 d_model=512 kernel_size=2 e_layers=1;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(96 192 336 720)
# NOTE: ECL settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek']";;
        192) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek']";;
        336) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek','SeasonOfYear']";;
        720) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3 time_feature_types="['HourOfDay','DayOfWeek','SeasonOfYear']";;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(96 192 336 720)
# NOTE: Traffic settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.01 batch_size=4 beta=0.3 d_model=512 kernel_size=5 e_layers=6;;
        192) lr=0.01 batch_size=4 beta=0.3 d_model=512 kernel_size=5 e_layers=1;;
        336) lr=0.01 batch_size=4 beta=0.3 d_model=512 kernel_size=5 e_layers=1;;
        720) lr=0.01 batch_size=16 beta=0.3 d_model=512 kernel_size=5 e_layers=3;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(96 192 336 720)
# NOTE: Weather settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.005 batch_size=128 beta=0.6 d_model=512 kernel_size=5 e_layers=3;;
        192) lr=0.005 batch_size=128 beta=0.6 d_model=256 kernel_size=2 e_layers=3;;
        336) lr=0.005 batch_size=128 beta=0.9 d_model=256 kernel_size=5 e_layers=3;;
        720) lr=0.005 batch_size=128 beta=0.9 d_model=128 kernel_size=5 e_layers=3;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(12 24 36 48)
# NOTE: PEMS03 settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        24) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        36) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        48) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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

pl_list=(12 24 36 48)
# NOTE: PEMS08 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        24) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        36) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
        48) lr=0.01 batch_size=16 beta=0.1 d_model=256 kernel_size=2 e_layers=3;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --time_feature_types $time_feature_types

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






wait