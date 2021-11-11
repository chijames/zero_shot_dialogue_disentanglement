cd entangled_response_selection/

# test different aux_weight
for i in 0.25
do
    python main.py --bert_model /path/to/bert-base-model/ --output_dir auxweight_$i/ --train_dir /path/to/dstc8_2 --gpu 0 --eval_batch_size 2 --use_pretrain --max_num_contexts 49 --fp16 --aux_weight $i --train_batch_size 8
done
