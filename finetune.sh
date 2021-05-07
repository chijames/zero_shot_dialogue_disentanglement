cd link_based_dialogue_disentanglement/
python main.py --bert_model ../entangled_response_selection/small_noaug_checkpoints/aux_pos_auxweight_0/ --output_dir null/ --train_dir link_no_time/ --max_num_contexts 50 --gold link_no_time/test/*anno* --gpu 1 --two_stage
#python main.py --bert_model ../../bert_model_small --output_dir null/ --train_dir link_no_time/ --max_num_contexts 50 --gold link_no_time/test/*anno* --gpu 1 --use_pretrain
