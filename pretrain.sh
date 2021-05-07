cd entangled_response_selection/

# test different aux_weight
# test the need of pos emb
for i in 0 1 3 5
do
    python main.py --bert_model ../../bert_model_small --output_dir small_noaug_checkpoints/aux_pos_auxweight_$i/ --train_dir ../../Poly-Encoder/dstc8_2/ --gpu 0 --eval_batch_size 2 --use_pretrain --max_num_contexts 49 --fp16 --aux_weight $i --use_pos_emb
    python main.py --bert_model ../../bert_model_small --output_dir small_noaug_checkpoints/aux_nopos_auxweight_$i/ --train_dir ../../Poly-Encoder/dstc8_2/ --gpu 0 --eval_batch_size 2 --use_pretrain --max_num_contexts 49 --fp16 --aux_weight $i
done

for i in 0 1 3 5
do
    python main.py --bert_model ../../bert_model_large --output_dir base_noaug_checkpoints/aux_pos_auxweight_$i/ --train_dir ../../Poly-Encoder/dstc8_2/ --gpu 0 --eval_batch_size 2 --use_pretrain --max_num_contexts 49 --fp16 --aux_weight $i --use_pos_emb
    python main.py --bert_model ../../bert_model_large --output_dir base_noaug_checkpoints/aux_nopos_auxweight_$i/ --train_dir ../../Poly-Encoder/dstc8_2/ --gpu 0 --eval_batch_size 2 --use_pretrain --max_num_contexts 49 --fp16 --aux_weight $i
done
