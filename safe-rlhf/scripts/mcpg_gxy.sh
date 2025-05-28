#!/usr/bin/env bash
#
# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
sed -i "s/slots=./slots=8/" /etc/mpi/hostfile
sed -i "s/slots=./slots=8/" /etc/mpi/mpi-hostfile

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

#all for kml Horovod
#modify PORT
PORT=$(grep -E '^#? *Port "[0-9]+"' /etc/ssh/ssh_config | awk '{gsub(/"/, "", $2); print $2}')
awk -v port="$PORT" '{if (/^#?Port/) {print "Port \"" port "\""} else {print}}' /etc/ssh/sshd_config > /tmp/sshd_config.tmp
mv /tmp/sshd_config.tmp /etc/ssh/sshd_config
/etc/init.d/ssh start


export NCCL_IB_DISABLE=0 \
export NCCL_IB_GID_INDEX=3 \
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4 \
export NCCL_DEBUG=WARN \
export NCCL_MIN_NCHANNELS=16 \
export NCCL_TOPO_DUMP_FILE=./topo \
export NCCL_DEBUG_SUBSYS=INIT \
export NCCL_NET_GDR_LEVEL=4 \
export UCX_NET_DEVICES=mlx5_bond_1:1 \


SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced"
MODEL_NAME_OR_PATH="/mmu_vcg_wjc_hdd/gaoxiaoyang/megatron_models/qwen7b_sft/ecom_dpo_8k_demo/global_step21734/"
MODEL_NAME_OR_PATH="/mmu_vcg_wjc_hdd/duyong/plms/QWen/Qwen2-7B-Instruct"
OUTPUT_DIR="${ROOT_DIR}/output/ecom_cot_mcpg_1105"
OUTPUT_DIR="/mmu_nlp_hdd/xiayu12/mcpg/book/cot"
unset HOSTFILE
ZERO_STAGE=2
OFFLOAD="none"
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--model_name_or_path=*)
			MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--hostfile)
			HOSTFILE="$1"
			shift
			;;
		--hostfile=*)
			HOSTFILE="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--offload)
			OFFLOAD="$1"
			shift
			;;
		--offload=*)
			OFFLOAD="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed "${DEEPSPEED_ARGS[@]}" \
	--hostfile /etc/mpi/hostfile \
	--module safe_rlhf.algorithms.mcpg \
	--train_datasets local-mcpg-dataset:1.0:/mmu_nlp_hdd/xiayu12/book_rlpf_cot.json \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 1024 \
	--trust_remote_code True \
	--epochs 1 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 1 \
	--gradient_checkpointing \
	--learning_rate 1e-6 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.05 \
	--seed 42 \
	--eval_strategy epoch \
	--scale_coeff 0.1 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project GXY-DPO \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True
