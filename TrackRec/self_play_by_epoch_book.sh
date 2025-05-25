#!/bin/bash

# 设置参数前缀
modelA_folder_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/data/book # 训练集文件夹名
modelA_infer_n_prefix=book1_tallrec_train        # 训练集文件json名 

# Generator 
modelA_save_infer_n_dir=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/book_infer_n_output  # infer n次保存的文件夹
model_A_path=/mmu_nlp_hdd/xiayu12/Qwen2.5_7B_Instruct         # model A的infer 模型路径 
# model_A_path=/share/ad/xiayu12/TALLRec/self_play/model_A/sft/output_step_3/checkpoint-24
model_A_save_infer_dir=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/book_infer_output     # model A为model Binfer训练数据的保存路径


modelB_folder_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/book_infer_n_output      # model B infer数据的文件夹
modelB_infer_n_prefix=train_cot_step                         # model B infer的文件名
modelB_save_infer_n_dir=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/V/book_infer_n_output  # model Binfer的保存文件夹
model_B_path=/mmu_nlp_hdd/xiayu12/Qwen2.5_7B_Instruct       # model B infer的模型路径

sample_folder_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/V/book_infer_n_output      # 采样model B infer的结果的输入文件夹
sample_prefix=infer_train_ans                               # 采样model B infer的结果的输入文件名
sample_dir=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/V/book_sample_preference_data         # 采样的保存路径
model_A_sample_dir=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/book_sft_data

# modelA_sft_model_path=/mmu_nlp_hdd/xiayu12/Qwen2.5_7B_Instruct # model A的训练模型的路径
# modelB_sft_model_path=/mmu_nlp_hdd/xiayu12/Qwen2.5_7B_Instruct # model B的训练模型的路径
model_B_base=/mmu_nlp_hdd/xiayu12/Qwen2.5_7B_Instruct

modelA_dpo_save_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/checkpoint/book
modelB_sft_save_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/V/checkpoint/book

# 设置迭代次数
iter=3  # 你可以根据需要更改这个值

cd /mmu_nlp_hdd/xiayu12/safe-rlhf_official/safe-rlhf
# 循环 iter 次
for ((i=1; i<=iter; i++))
do
    echo "Iteration $i of $iter"
    
    # 根据迭代次数和前缀修改参数
    # param="${prefix}_iter${i}"
    
    # 执行 a.py，传递参数
    echo "model_A_infer_n_cot"
    conda activate base
    python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/dy_multi_gpu_vllm_infer_n_seed_cot.py \
        --folder_path "$modelA_folder_path" \
        --path_prefix "$modelA_infer_n_prefix" \
        --save_infer_n_dir "$modelA_save_infer_n_dir" \
        --iter $i \
        --model_path "$model_A_path"

    # 执行model B的infer
    echo "model_B_infer_n_ans"
    conda activate base
    python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/V/dy_multi_gpu_vllm_infer_ans.py \
        --folder_path "$modelB_folder_path" \
        --path_prefix "${modelB_infer_n_prefix}_${i}" \
        --save_infer_n_dir "$modelB_save_infer_n_dir" \
        --iter $i \
        --model_path "$model_B_path"

    # 执行sample
    echo "sample preference cot"
    python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/V/sample_preference_data.py \
        --folder_path "$sample_folder_path" \
        --path_prefix "${sample_prefix}_${i}" \
        --iter $i \
        --sample_dir "${sample_dir}"
    
    # exit 0

    # 执行 modelA 的sft
    conda deactivate
    bash /mmu_nlp_hdd/xiayu12/safe-rlhf_official/safe-rlhf/scripts/sdpo_gxy.sh $model_A_path "$modelA_dpo_save_path/cot_sdpo_epoch_$i" "$sample_dir/train_sdpo_step_$i.json" 
    model_A_path="$modelA_dpo_save_path/cot_sdpo_epoch_$i" 

    # 训练完的model A为model B生成训练数据
    echo "infer trained model A for data to train model B"
    conda activate base
    python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/multi_gpu_vllm_infer_cot.py \
    --folder_path "$modelA_folder_path" \
    --path_prefix "$modelA_infer_n_prefix" \
    --save_infer_n_dir "$model_A_save_infer_dir" \
    --iter $i \
    --model_path "$model_A_path" 

    python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/sample_rec_data.py \
    --folder_path "$model_A_save_infer_dir" \
    --path_prefix "infer_model_A_step_${i}_" \
    --iter $i\
    --sample_dir "$model_A_sample_dir"

    # 执行modelB的sft
    echo "finetuing model B"
    conda deactivate
    bash /mmu_nlp_hdd/xiayu12/safe-rlhf_official/safe-rlhf/scripts/sft_gxy.sh $model_B_base "$modelB_sft_save_path/cot_sft_epoch_$i" "$model_A_sample_dir/train_sft_step_$i.json" model_A_path="$modelA_dpo_save_path/cot_sdpo_epoch_$i" 
    model_B_path="$modelB_sft_save_path/cot_sft_epoch_$i"
 

    echo "Completed iteration $i"
done

echo "Self-play completed after $iter iterations."
