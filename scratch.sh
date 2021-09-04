cd link_based_dialogue_disentanglement/

for i in 12345 1 2
do
    for j in 1.0
    do
        python main.py --bert_model ~/bert_model_small --output_dir camera_ready_"$i"/scratch --train_dir link_no_time/ --max_num_contexts 50 --test_gold link_no_time/test/*anno* --gpu 0 --use_pretrain --dev_gold link_no_time/dev/*anno* --data_percent $j --seed $i
    done
done
