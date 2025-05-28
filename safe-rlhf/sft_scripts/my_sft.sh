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

###train_data
#TRAIN_DATA_PATH=/mmu_nlp_ssd/wanglei16/llm/prepare_data/dataset/landingpage/landingpage_v2.2_allcate_qwen.json
#TRAIN_DATA_PATH=/mmu_nlp_ssd/lizhiwei05/project/app/sft/photo视频素材大模型训练集_sft.jsonl
#TRAIN_DATA_PATH=/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/reco2-v1/data/book1_reco_wirlpf_distill_coldstart_sft_train1.json
#TRAIN_DATA_PATH=/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/reco2-v1/data/our_reco_wirlpf_distill_coldstart_sft_train.json
#TRAIN_DATA_PATH=/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/reco2-v1/data/movielen1_recowirlpf_qw7b_sft_train.json
TRAIN_DATA_PATH=/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/reco2-v1/data/book1_recowirlpf_qw7b_sft_train.json
run_name=book1_recowirlpf_xx
MAX_LEN=1024
epoch=1

global_batch_size=128 # 必须能被 $num_nodes * $num_gpus * $train_batch_size 整除
train_batch_size=4

###hyper_parameters
num_nodes=1
#######################################################################################################################
num_gpus=4
LR=1e-5
###pretrain_model
MODEL_NAME_OR_PATH=/mmu_vcg_wjc_hdd/duyong/plms/QWen/Qwen2-7B-Instruct
arch=qwen2-7b

###exp_name
exp_name=qwen2
REAL_TRAIN_DATA_PATH=/mmu_nlp_hdd/xiayu12/mcpg/movie/infer_train/movie_cot.json

###output
OUTPUT_DIR=/mmu_nlp_hdd/xiayu12/mcpg/movie/cot_sft
LOG_DIR=${OUTPUT_DIR}

# *********************************************************************************************************************
# *********************************************************************************************************************

uri=/nlp_group/mlflow_monitor/sft

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
echo ${np}
echo ${Port}
set -x

mpirun --allow-run-as-root -np $np \
    -mca plm_rsh_args "-p ${Port}"  \
    -hostfile ${hostfile} \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x MPI_THREAD_SINGLE=1 \
    -bind-to none \
    -map-by slot \
    -mca btl_tcp_if_include eth04 \
    -mca btl_openib_allow_ib true \
    --mca btl tcp,self \
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_MIN_NCHANNELS=16 \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x NCCL_DEBUG=WARN \
python -m safe_rlhf.finetune \
    --train_datasets kuaiyi-sft-dataset:1.0:${REAL_TRAIN_DATA_PATH} \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --max_length ${MAX_LEN} \
    --trust_remote_code True \
    --epochs ${epoch} \
    --per_device_train_batch_size ${train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --gradient_checkpointing \
	--learning_rate ${LR} \
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
