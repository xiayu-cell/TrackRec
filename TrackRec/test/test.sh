#!/bin/bash

iter=1
# 设置参数前缀
model_A_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/G/checkpoint/movie/cot_sdpo_epoch_$iter
model_B_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/V/checkpoint/movie/cot_sft_epoch_$iter
test_data_prefix=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/data/movie
test_data_path=movielen1_tallrec_test
save_infer_n_dir=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/test/G

echo "start infer model A"
conda activate base
python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/test/multi_gpu_vllm_infer_cot.py \
        --folder_path "$test_data_prefix" \
        --path_prefix "$test_data_path" \
        --save_infer_n_dir "$save_infer_n_dir" \
        --iter $iter \
        --model_path "$model_A_path"
if [ $? -ne 0 ]; then
    echo "Error executing a.py"
    exit 1
fi
# 执行model B的infer
echo "start infer model B to get ans"
folder_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/test/G
path_prefix="model_A_cot_iter_${iter}"
save_infer_n_dir=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/test/V
python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/test/multi_gpu_vllm_infer_ctr_only_ans.py \
        --folder_path "$folder_path" \
        --path_prefix "$path_prefix" \
        --save_infer_n_dir "$save_infer_n_dir" \
        --iter $iter \
        --model_path "$model_B_path"

echo "caculate test AUC and ACC"
folder_path=/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/test/V
path_prefix="model_B_iter_${iter}"
python /mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/test/cal_test_only_ans.py \
        --folder_path "$folder_path" \
        --path_prefix "$path_prefix" \
        --iter $iter
    

echo "test completed for $iter iterations."
