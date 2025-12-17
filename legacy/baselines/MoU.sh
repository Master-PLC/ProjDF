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
des='MoU'

model_name=MoU
auxi_mode=rfft
datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Traffic PEMS03 PEMS08)
# datasets=(ETTh1)


# hyper-parameters
dst=ETTh1

train_epochs=100
patience=20
test_batch_size=1
lambda=1.0
lr=0.0002
batch_size=512
lradj=TST
d_model=64
d_ff=128
n_heads=4
entype=mof
ltencoder=mfca
postype=w
e_layers=1
dropout=0.1
fc_dropout=0.1
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTh1 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 dps="[0.1,0.1,0.1,0.0,0.1]";;
        192) lr=0.0001 dps="[0.1,0.1,0.1,0.0,0.1]";;
        336) lr=0.0001 dps="[0.1,0.1,0.1,0.0,0.1]";;
        720) lr=0.00007 dps="[0.1,0.3,0.3,0.3,0.3]";;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





# hyper-parameters
dst=ETTh2

train_epochs=100
patience=20
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=512
lradj=TST
d_model=64
d_ff=128
n_heads=4
entype=mof
ltencoder=mfca
postype=w
e_layers=1
dropout=0.2
fc_dropout=0.1
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTh2 settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 dps="[0.2,0.2,0.2,0.0,0.2]";;
        192) lr=0.0001 dps="[0.2,0.2,0.2,0.0,0.2]";;
        336) lr=0.0001 dps="[0.2,0.2,0.2,0.0,0.2]";;
        720) lr=0.0001 dps="[0.2,0.2,0.2,0.0,0.2]";;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=ETTm1

train_epochs=100
patience=20
test_batch_size=1
lambda=1.0
lr=0.0002
batch_size=512
lradj=TST
d_model=128
d_ff=256
n_heads=16
entype=dyconv
ltencoder=mfca
postype=w
e_layers=1
dropout=0.2
fc_dropout=0.2
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=8
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTm1 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.00007 dps="[0.2,0.2,0.2,0.0,0.2]";;
        192) lr=0.00007 dps="[0.2,0.2,0.2,0.0,0.2]";;
        336) lr=0.00007 dps="[0.2,0.2,0.2,0.0,0.2]";;
        720) lr=0.00007 dps="[0.2,0.2,0.2,0.0,0.2]";;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








# hyper-parameters
dst=ETTm2

train_epochs=100
patience=20
test_batch_size=1
lambda=1.0
lr=0.001
batch_size=512
lradj=TST
d_model=128
d_ff=256
n_heads=16
entype=mof
ltencoder=mfca
postype=w
e_layers=1
dropout=0.1
fc_dropout=0.1
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTm2 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.00002 dps="[0.2,0.0,0.0,0.0,0.1]";;
        192) lr=0.00002 dps="[0.2,0.0,0.0,0.0,0.1]";;
        336) lr=0.00002 dps="[0.2,0.0,0.0,0.0,0.1]";;
        720) lr=0.00002 dps="[0.2,0.0,0.0,0.0,0.1]";;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=ECL

train_epochs=100
patience=10
test_batch_size=1
lambda=1.0
lr=0.0005
batch_size=96
lradj=TST
d_model=128
d_ff=256
n_heads=16
entype=mof
ltencoder=mamba
postype=w
e_layers=2
dropout=0.2
fc_dropout=0.2
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(96 192 336 720)
# NOTE: ECL settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=96;;
        192) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        336) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        720) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=Traffic

train_epochs=100
patience=10
test_batch_size=1
lambda=1.0
lr=0.0005
batch_size=8
lradj=TST
d_model=128
d_ff=256
n_heads=16
entype=mof
ltencoder=mamba
postype=w
e_layers=2
dropout=0.2
fc_dropout=0.2
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(96 192 336 720)
# NOTE: Traffic settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        192) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        336) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        720) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=Weather

train_epochs=100
patience=20
test_batch_size=1
lambda=1.0
lr=0.0005
batch_size=256
lradj=TST
d_model=128
d_ff=256
n_heads=16
entype=mof
ltencoder=mfca
postype=w
e_layers=1
dropout=0.2
fc_dropout=0.2
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=4
d_state=21
rerun=0

pl_list=(96 192 336 720)
# NOTE: Weather settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 dps="[0.1,0.2,0.2,0.0,0.2]" d_state=21;;
        192) lr=0.0001 dps="[0.1,0.2,0.2,0.0,0.2]" d_state=21;;
        336) lr=0.0001 dps="[0.1,0.2,0.2,0.0,0.2]" d_state=16;;
        720) lr=0.0001 dps="[0.1,0.2,0.2,0.0,0.2]" d_state=16;;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=PEMS03

train_epochs=100
patience=20
test_batch_size=1
lambda=1.0
lr=0.0005
batch_size=96
lradj=TST
d_model=128
d_ff=256
n_heads=16
entype=mof
ltencoder=mamba
postype=w
e_layers=2
dropout=0.2
fc_dropout=0.2
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(12 24 36 48)
# NOTE: PEMS03 settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.0001 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=96;;
        24) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        36) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        48) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








# hyper-parameters
dst=PEMS08

train_epochs=100
patience=20
test_batch_size=1
lambda=1.0
lr=0.001
batch_size=96
lradj=TST
d_model=128
d_ff=256
n_heads=16
entype=mof
ltencoder=mamba
postype=w
e_layers=2
dropout=0.2
fc_dropout=0.2
head_dropout=0.0
patch_len=16
stride=8
kernel_size=25
top_k=2
num_x=4
conv_stride=16
conv_kernel_size=16
expand=2
d_state=21
rerun=0

pl_list=(12 24 36 48)
# NOTE: PEMS08 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.0001 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=96;;
        24) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        36) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
        48) lr=0.00012 dps="[0.1,0.2,0.2,0.2,0.3]" batch_size=8;;
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --d_model $d_model \
            --d_ff $d_ff \
            --n_heads $n_heads \
            --dropout $dropout \
            --entype $entype \
            --ltencoder $ltencoder \
            --postype $postype \
            --e_layers $e_layers \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --patch_len $patch_len \
            --stride $stride \
            --kernel_size $kernel_size \
            --top_k $top_k \
            --dps $dps \
            --num_x $num_x \
            --conv_stride $conv_stride \
            --conv_kernel_size $conv_kernel_size \
            --expand $expand \
            --d_state $d_state

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






wait