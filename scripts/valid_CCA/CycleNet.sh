#!/bin/bash
MAX_JOBS=16
GPUS=(4 5 6 7)
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
EXP_NAME=finetune_loss
seed=2023
des='CycleNet'

rank_ratio=1.0
align_type=1
pca_dim=D
proj_init=cca
auxi_mode=cca
auxi_loss=None
ind=0

model_name=CycleNet
datasets=(ETTm2)



# hyper-parameters
dst=ETTh1
pl_list=(96 192 336 720)

alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=15
use_revin=1
model_type=linear
cycle=24
bs_list=(32 128 256)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)

rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh1_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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





# hyper-parameters
dst=ETTh2
pl_list=(96 192 336 720)


alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=15
use_revin=1
model_type=linear
cycle=24
bs_list=(32 128 256)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)

rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh2_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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







# hyper-parameters
dst=ETTm1
pl_list=(96 192 336 720)

alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=15
use_revin=1
model_type=linear
cycle=96
bs_list=(32 128 256)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)

rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm1_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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







# hyper-parameters
dst=ETTm2

use_revin=1
model_type=linear
cycle=96
test_batch_size=1
fixed_epoch=0
rerun=0


pl_list=(96 192 336 720)
alpha_list=(0.001)
reg_lambda_list=(0.01)
lr_list=(0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_step_list=(500)
lradj_list=(type3)
train_epochs=100
patience=30
bs_list=(128)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)
# NOTE: ETTm2 settings

# NOTE: ETTm2 running
for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm2_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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






# hyper-parameters
dst=ECL
pl_list=(96 192 336 720)

alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=15
use_revin=1
model_type=linear
cycle=168
bs_list=(32 128)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)

rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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





# hyper-parameters
dst=Traffic
pl_list=(96 192 336 720)

alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=10
use_revin=1
model_type=linear
cycle=168
bs_list=(32 128)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)

rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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






# hyper-parameters
dst=Weather
pl_list=(96 192 336 720)

alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=15
model_type=linear
cycle=144
bs_list=(32 128 256)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)
rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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





# hyper-parameters
dst=PEMS03
pl_list=(12 24 36 48)

alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=15
use_revin=0
model_type=mlp
cycle=288
bs_list=(32 128 256)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)

rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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








# hyper-parameters
dst=PEMS08
pl_list=(12 24 36 48)

alpha_list=(0.005 0.01 0.02 0.05)
reg_lambda_list=(0.0)
lr_list=(0.005 0.002 0.001)
lx_list=(0 1)
ly_list=(0 1)
fixed_epoch=0
fixed_step_list=(-1 300 400 500)
lradj_list=(TST type3)
train_epochs=100
patience=15
use_revin=0
model_type=mlp
cycle=288
bs_list=(32 128 256)
rank_ratio_list=(1.0)
pre_norm_list=(0)
align_types=(1 5)
identity_directions=(right)
cca_types=(svd)
proj_init_list=(cca)
auxi_types=(1)

rerun=0


for lr in ${lr_list[@]}; do
lr_inner=$(echo "scale=10; $lr / 2" | bc)
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')
for rank_ratio in ${rank_ratio_list[@]}; do
for pre_norm in ${pre_norm_list[@]}; do
for batch_size in ${bs_list[@]}; do
for reg_lambda in ${reg_lambda_list[@]}; do
for lradj in ${lradj_list[@]}; do
for fixed_step in ${fixed_step_list[@]}; do
for lx in ${lx_list[@]}; do
for ly in ${ly_list[@]}; do
if [[ $lx -eq 0 && $ly -eq 0 ]]; then
    continue
fi
for align_type in ${align_types[@]}; do
for identity_direction in ${identity_directions[@]}; do
for auxi_type in ${auxi_types[@]}; do
for proj_init in ${proj_init_list[@]}; do
for cca_type in ${cca_types[@]}; do
for alpha in ${alpha_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=1.0
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lradj}_${train_epochs}_${patience}_${batch_size}_${rank_ratio}_${align_type}_${pca_dim}_${ind}_${fixed_epoch}_${fixed_step}_${lx}_${ly}_${reg_lambda}_${proj_init}_${pre_norm}_${identity_direction}_${auxi_mode}_${auxi_loss}_${auxi_type}_${cca_type}
    OUTPUT_DIR="./results_CCA/${EXP_NAME}/${JOB_NAME}"

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

    proj_dir="./projections/CCA/${dst}/${pl}"
    mkdir -p "${proj_dir}"


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_cca_cycle_loss \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS_CCA_Cycle \
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
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --learn_x_proj $lx \
            --learn_y_proj $ly \
            --rank_ratio $rank_ratio \
            --align_type $align_type \
            --pca_dim $pca_dim \
            --fixed_epoch $fixed_epoch \
            --proj_init $proj_init \
            --fixed_step $fixed_step \
            --inner_lr $lr_inner \
            --individual ${ind} \
            --reg_lambda ${reg_lambda} \
            --pre_norm ${pre_norm} \
            --identity_direction ${identity_direction} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --load_from_disk ${proj_dir} \
            --auxi_type ${auxi_type} \
            --cca_type ${cca_type}

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




wait