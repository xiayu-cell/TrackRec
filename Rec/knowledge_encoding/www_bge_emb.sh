for data_type in "train" "test"
do
  for model_type in "sdpo3" "distill" "sdpo1" "sdpo2"
  do
    echo $model_type $data_type
    python www_bge_emb.py --model-type $model_type --data-type $data_type
  done
done
