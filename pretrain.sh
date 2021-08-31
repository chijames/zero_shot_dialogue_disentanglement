cd entangled_response_selection/

# test different aux_weight
for i in 0.0 0.25 0.5 0.75 1.0
do
    python main.py --bert_model ~/bert_model_small --output_dir camera_ready/auxweight_$i/ --train_dir /usr1/tachungc/Poly-Encoder/dstc8_2/ --gpu 1 --eval_batch_size 2 --use_pretrain --max_num_contexts 49 --fp16 --aux_weight $i
done
