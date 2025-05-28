#!/bin/bash
# -*- coding: utf-8 -*-

sed -i "s/slots=1/slots=8/" /etc/mpi/hostfile
sed -i "s/slots=1/slots=8/" /etc/mpi/mpi-hostfile

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

# **************************************************** 训练配置 ********************************************************
# *********************************************************************************************************************

MODEL_NAME_OR_PATH=/mmu_nlp_hdd/wangbo27/kml_output/sft/Qwen2-7B-stage1-v1/global_step_1814/hf_bin

TRAIN_DATA_PATH=/nlp_group/linlei/sft_case_fix/qwen7b_train/sft_stage2_for-linlei_v6.json

run_name=Qwen2_7B_stage2_train_for-linlei_v6

global_batch_size=512 # 必须能被 $num_nodes * $num_gpus * $train_batch_size 整除
train_batch_size=4
num_nodes=8
num_gpus=8
epoch=5

exp_name=qwen2
arch=qwen2-7b

OUTPUT_DIR=/mmu_nlp_hdd/linlei/megatron_models/qwen7b_sft/${run_name}

# *********************************************************************************************************************
# *********************************************************************************************************************

uri=/nlp_group/mlflow_monitor/sft

LOG_DIR=/mmu_nlp_hdd/linlei/megatron_models/qwen7b_sft/${run_name}

REAL_TRAIN_DATA_PATH=data/${run_name}_train.json

gradient_accumulation_steps=$(( $global_batch_size / $num_nodes / $num_gpus / $train_batch_size ))
sample_num=$(cat $TRAIN_DATA_PATH |wc -l)
epoch_step=$(( $sample_num / $num_gpus / $num_nodes / $train_batch_size ))
eval_step=$(( $epoch_step / 1 ))

ZERO_STAGE=3
OFFLOAD="none" # "optimizer"

rm -rf ${uri}/${exp_name}/${exp_name}_${run_name}
mkdir -p ${uri}/${exp_name}/${exp_name}_${run_name}
ln -s ${OUTPUT_DIR} ${uri}/${exp_name}/${exp_name}_${run_name}/artifacts

shuf $TRAIN_DATA_PATH > $REAL_TRAIN_DATA_PATH

echo "Epoch steps number: $epoch_step"
echo "Eval steps number: $eval_step"
echo "gradient_accumulation_steps: $gradient_accumulation_steps"

mkdir -p "${OUTPUT_DIR}"

hostfile="/etc/mpi/hostfile"
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
echo $np
mpirun --allow-run-as-root -np $np \
    -mca plm_rsh_args "-p ${Port}"  \
    -hostfile ${hostfile} \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x MPI_THREAD_SINGLE=1 \
    -bind-to none \
    -map-by slot \
    -mca btl_tcp_if_include bond0 \
    -mca btl_openib_allow_ib false \
    --mca btl tcp,self \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_MIN_NCHANNELS=16 \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x NCCL_DEBUG=WARN \
python -m safe_rlhf.finetune \
    --train_datasets kuaiyi-sft-dataset:1.0:${REAL_TRAIN_DATA_PATH} \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --max_length 2048 \
    --trust_remote_code True \
    --epochs ${epoch} \
    --per_device_train_batch_size ${train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --gradient_checkpointing \
	--learning_rate 1e-5 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
    --seed 122 \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --log_project SFT \
    --zero_stage "${ZERO_STAGE}" \
    --offload "${OFFLOAD}" \
    --bf16 True \
    --tf32 True \
    --exp_name ${exp_name} \
    --run_name ${run_name} \
    --uri ${uri} \
    --save_16bit \
    --arch ${arch} \
    --need_save \
    --save_strategy epoch