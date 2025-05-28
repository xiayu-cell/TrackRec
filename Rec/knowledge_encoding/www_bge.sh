#!/bin/bash

for i in `seq 0 $((6-1))` ; do (
    	sleep $((i*5))
	python www_bge_emb_pca_128.py --item-out-path ../data/ecom_www/proc_data/test_1_${i}_item.emb --user-out-path ../data/ecom_www/proc_data/test_1_${i}_user.emb --gpu ${i} --prefix kar_test_model_A_step_1_1_0${i}
) & 
done
wait
