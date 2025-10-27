#!/bin/bash
MAX_JOBS=8
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
EXP_NAME=finetune
seed=2023
des='Fredformer'

model_name=Fredformer
# datasets=(ETTh1 ETTh2 ETTm1 ETTm2 ECL Weather)
datasets=(Weather)
# datasets=(ETTh1 ETTh2)
# datasets=(ETTh1)


normalize=1
auxi_loss=None
ot_type=upper_bound
mask_factor=0.0
auxi_mode=fft_ot

# hyper-parameters
dst=ETTh1

lambda=1.0
lr=0.0001
lradj=type3
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.1 0.2)
lr_list=(0.0001 0.0002 0.0005)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type3)
joint_forecast_list=(1)
bs_list=(128)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        96)  cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        192) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --d_model $d_model \
            --d_ff 128 \
            --dropout 0.3 \
            --fc_dropout 0.3 \
            --patch_len 4 \
            --stride 4 \
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
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim\
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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

lambda=1.0
lr=0.0001
lradj=type3
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.1 0.02)
lr_list=(0.0005 0.0002 0.0001)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type3)
joint_forecast_list=(1)
bs_list=(128)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        96)  cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        192) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 4 \
            --d_model $d_model \
            --d_ff 128 \
            --dropout 0.3 \
            --fc_dropout 0.3 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
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
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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

lambda=1.0
lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.1 0.8 0.9)
lr_list=(0.0005 0.0002 0.0001)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(TST)
joint_forecast_list=(1)
bs_list=(128)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        96)  cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        192) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.4 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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

lambda=1.0
lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.2 0.3 0.5)
lr_list=(0.0005 0.0002 0.0001)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(TST)
joint_forecast_list=(1)
bs_list=(128)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        96)  cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        192) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.4 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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

lambda=1.0
lr=0.01
lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.01 0.05)
lr_list=(0.005 0.002 0.01)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(TST)
joint_forecast_list=(1)
bs_list=(32)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        96)  cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        192) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        336) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        720) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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
pl_list=(96 192 336 720)

lambda=1.0


lr=0.01
lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1
rerun=0

pl_list=(96 192 336 720)
alpha_list=(0.1 0.2)
lr_list=(0.0005 0.0002 0.001)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32 64)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        96)  cf_dim=512 cf_depth=3 cf_heads=8 cf_mlp=512 cf_head_dim=32 d_model=256;;
        192) cf_dim=512 cf_depth=3 cf_heads=8 cf_mlp=512 cf_head_dim=32 d_model=256;;
        336) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=48 d_model=256;;
        720) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=48 d_model=256;;
    esac
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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

lambda=1.0
lr=0.001
lradj=type3
train_epochs=100
patience=5
batch_size=128
test_batch_size=1
rerun=0

# pl_list=(96 192 336 720)
# alpha_list=(0.01 0.005)
# lr_list=(0.0005 0.0002 0.001)
# distance_list=(wasserstein_empirical_per_dim)
# lradj_list=(type3)
# joint_forecast_list=(1)
# bs_list=(128)
# eps_list=(1e-9)
# reg_sk_list=(0.005)
# vw_list=(0.002)


pl_list=(96 192 336 720)
alpha_list=(0.002 0.001 0.0005)
lr_list=(0.001 0.002 0.0015)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type3)
joint_forecast_list=(1)
bs_list=(128)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        96)  cf_dim=32 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=8;;
        192) cf_dim=32 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=8;;
        336) cf_dim=16 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=4;;
        720) cf_dim=16 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=4;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 16 \
            --stride 16 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
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
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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

lambda=1.0
lr=0.01
lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1
rerun=0

pl_list=(12 24 36 48)
alpha_list=(0.1 0.2)
lr_list=(0.0005 0.0002 0.001)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32 64)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        12)  cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        24) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        36) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        48) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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

lambda=1.0
lr=0.01
lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1
rerun=0

pl_list=(12 24 36 48)
alpha_list=(0.1 0.2)
lr_list=(0.0005 0.0002 0.001)
distance_list=(wasserstein_empirical_per_dim)
lradj_list=(type1)
joint_forecast_list=(1)
bs_list=(32 64)
eps_list=(1e-9)
reg_sk_list=(0.005)
vw_list=(0.002)
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

    case $pl in
        12)  cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        24) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        36) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        48) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
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
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
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