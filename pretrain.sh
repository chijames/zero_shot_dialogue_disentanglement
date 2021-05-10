cd entangled_response_selection/

# test different aux_weight
for i in 0.0 0.25 0.5 0.75 1.0
do
    python main.py --bert_model ../../bert_model_small --output_dir small_noaug_checkpoints/auxweight_$i/ --train_dir ../../Poly-Encoder/dstc8_2/ --gpu 0 --eval_batch_size 2 --use_pretrain --max_num_contexts 49 --fp16 --aux_weight $i
done
