export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM="true"


## to change
dataset="book1_distill_rlpf_gpt4o"
max_seq_length=1024
echo $dataset
echo $max_seq_length
train_data="train"
valid_data="valid"
train_fn="data/${dataset}_${train_data}.json"
eval_fn="data/${dataset}_${valid_data}.json"

echo "train data: "$train_fn
echo "valid data: "$eval_fn
gpu_num=2
epoch=1
lr=1e-5


base_model="Qwen2-7B-Instruct"
model_path=$base_model
echo $model_path
sftmode="full"
gpu_group="0,1,2,3,4,5,6,7"
per_device_bs=4
grad_accum_step=4
bs=$(($gpu_num*$per_device_bs*$grad_accum_step))
output_model=${dataset}_${base_model}_lr${lr}_bs${bs}_epoch${epoch}_${sftmode}_${train_data}
output_model_path="/model/"${output_model}

accelerate launch \
    --config_file=./yaml/deepspeed_zero2.yaml \
    --num_processes ${gpu_num} \
    sft.py \
    --model_name_or_path ${model_path} \
    --jsonl_path ${train_fn}\
    --eval_path ${eval_fn} \
    --evaluation_strategy epoch \
    --max_seq_length ${max_seq_length} \
    --learning_rate ${lr} \
    --per_device_train_batch_size ${per_device_bs} \
    --gradient_accumulation_steps ${grad_accum_step} \
    --num_train_epochs ${epoch} \
    --output_dir ${output_model_path} \
    --logging_steps 25 \
    --save_strategy epoch\
    --max_steps -1 \
    --bf16 
