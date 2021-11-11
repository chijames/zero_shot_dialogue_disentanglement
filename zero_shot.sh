cd link_based_dialogue_disentanglement/

# test different aux_weight
for i in 0.25
do
    python main.py --bert_model /path/to/bert-model/ --output_dir ../entangled_response_selection/auxweight_0.25/ --train_dir link_no_time/ --max_num_contexts 50 --test_gold link_no_time/test/*anno* --gpu 0 --eval --dev_gold link_no_time/dev/*anno*
done
