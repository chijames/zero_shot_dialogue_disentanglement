cd link_based_dialogue_disentanglement/

rm -rf auxweight_0.25/
cp -r ../entangled_response_selection/auxweight_0.25/ ./
for j in 0.01 0.1 1
do
    python main.py --bert_model auxweight_0.25 --output_dir finetuned_auxweight_0.25 --train_dir link_no_time/ --max_num_contexts 50 --test_gold link_no_time/test/*anno* --gpu 0 --two_stage --dev_gold link_no_time/dev/*anno* --data_percent $j
done
