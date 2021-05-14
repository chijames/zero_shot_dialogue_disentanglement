import os
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from task_4_evaluation import *

from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset import SelectionDataset
from model import CrossEncoder

from sklearn.metrics import label_ranking_average_precision_score

import logging
logging.basicConfig(level=logging.ERROR)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def eval_running_model(dataloader, test=False):
    model.eval()
    preds = []
    for step, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch, labels_batch = batch
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch)
        preds += torch.argmax(logits, 1).data.cpu().numpy().tolist()
    '''
    with open('link/map_res/{}.json'.format(args.max_num_contexts), 'w') as outfile:
        json.dump(preds, outfile)
    exit()
    '''
    if 'test' in args.gold[0]:
        ids = open(os.path.join(args.train_dir, 'test_ids.txt')).read().splitlines()
        write_file = 'merged_test.txt'
    else:
        ids = open(os.path.join(args.train_dir, 'dev_ids.txt')).read().splitlines()
        write_file = 'merged_dev.txt'
    out = open(write_file, 'w')
    for id, pred in zip(ids, preds):
        filename, id = id.split(' ')
        pred = int(pred)
        id = int(id)
        out.write('{}.annotation.txt:{} {} -\n'.format(filename, id-args.max_num_contexts+pred+1, id))
    out.close()
    # read in evaluation data
    gold, gpoints, gedges = read_data(args.gold)
    auto, apoints, aedges = read_data([write_file])
    issue = False
    for filename in auto:
        if filename not in gold:
            print("Gold is missing file {}".format(filename), file=sys.stderr)
            issue = True
    for filename in gold:
        if filename not in auto:
            print("Auto is missing file {}".format(filename), file=sys.stderr)
            issue = True
    if issue:
        sys.exit(0)
    if len(apoints.symmetric_difference(gpoints)) != 0:
        print(apoints.difference(gpoints))
        print(gpoints.difference(apoints))
        raise Exception("Set of lines does not match: {}".format(apoints.symmetric_difference(gpoints)))

    ret = {}
    contingency, row_sums, col_sums = clusters_to_contingency(gold, auto)
    ret['vi'] = variation_of_information(contingency, row_sums, col_sums)
    ret['ari'] = adjusted_rand_index(contingency, row_sums, col_sums)
    ret['em'] = exact_match(gold, auto)
    ret['ls'] = link_score(gedges, aedges)

    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
    parser.add_argument("--two_stage", action="store_true", help="provide no struct model in bert_model")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--train_dir", default='data/ubuntu_data', type=str)

    parser.add_argument("--use_pretrain", action="store_true")

    parser.add_argument("--data_percent", type=float, default=1.0)
    parser.add_argument("--max_contexts_length", default=32, type=int)
    parser.add_argument("--max_num_contexts", default=15, type=int)
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=100, type=int, help="Log frequency")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=100, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                                            help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gold', help='File(s) containing the gold clusters, one per line. If a line contains a ":" the start is considered a filename', required=True, nargs="+")
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    set_seed(args)

    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizerFast, BertModel),
    }
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

    ## init dataset and bert model
    tokenizer = TokenizerClass.from_pretrained(args.bert_model, do_lower_case=True, clean_text=False)

    print('=' * 80)
    print('Train dir:', args.train_dir)
    print('Output dir:', args.output_dir)
    print('=' * 80)

    if not args.eval:
        train_dataset = SelectionDataset(os.path.join(args.train_dir, 'train.txt'), args, tokenizer, sample_cnt=args.data_percent)
        val_dataset = SelectionDataset(os.path.join(args.train_dir, 'test.txt'), args, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.batchify_join_str, shuffle=True, num_workers=1)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    else: # test
        val_dataset = SelectionDataset(os.path.join(args.train_dir, 'test.txt'), args, tokenizer)

    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=val_dataset.batchify_join_str, shuffle=False, num_workers=1)


    epoch_start = 1
    global_step = 0
    best_eval_loss = 0
    best_test_loss = 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    shutil.copyfile(os.path.join(args.bert_model, 'vocab.txt'), os.path.join(args.output_dir, 'vocab.txt'))
    shutil.copyfile(os.path.join(args.bert_model, 'config.json'), os.path.join(args.output_dir, 'config.json'))
    log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8')
    print (args, file=log_wf)

    state_save_path = os.path.join(args.output_dir, 'pytorch_model.bin1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    ## build BERT encoder
    ########################################
    bert_config = ConfigClass.from_json_file(os.path.join(args.bert_model, 'config.json'))
    bert = BertModelClass(bert_config)
    if args.use_pretrain and not args.eval:
        for filename in os.listdir(args.bert_model):
            if 'pytorch_model.bin' in filename:
                previous_model_file = os.path.join(args.bert_model, filename)
                break
        print('Loading parameters from', previous_model_file)
        log_wf.write('Loading parameters from %s' % previous_model_file + '\n')
        model_state_dict = torch.load(previous_model_file, map_location="cpu")
        try:
            bert = BertModelClass.from_pretrained(pretrained_model_name_or_path=None, config=bert_config, state_dict=model_state_dict)
        except:
            bert.resize_token_embeddings(len(tokenizer)) 
            bert = BertModelClass.from_pretrained(pretrained_model_name_or_path=None, config=bert_config, state_dict=model_state_dict)
            
        del model_state_dict
    bert.resize_token_embeddings(len(tokenizer)) 

    model = CrossEncoder(bert_config, bert=bert, max_num_contexts=args.max_num_contexts)
    #tokenizer.add_tokens(['\n'], special_tokens=True)
    #model.resize_token_embeddings(len(tokenizer)) 
    if args.two_stage:
        previous_model_file = os.path.join(args.bert_model, 'pytorch_model.bin1')
        print('Loading parameters from', previous_model_file)
        log_wf.write('Loading parameters from %s' % previous_model_file + '\n')
        model_state_dict = torch.load(previous_model_file)
        model.load_state_dict(model_state_dict, strict=False)
        del model_state_dict
    model.to(device)
    
    if args.eval:
        print('Loading parameters from', state_save_path)
        model_state_dict = torch.load(state_save_path)
        if len(set(model.state_dict().keys()) - set(model_state_dict.keys())) != 0:
            print ('Checkpoint ERROR! Some of the variables are not properly loaded with pretrained weights. Check tensor names.')
            exit()
        model.load_state_dict(model_state_dict, strict=False)
        test_result = eval_running_model(val_dataloader, test=True)
        print (test_result)
        exit()
        
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total*0.1), num_training_steps=t_total
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    print_freq = args.print_freq//args.gradient_accumulation_steps
    eval_freq = min(len(train_dataloader) // 2, 1000)
    eval_freq = eval_freq//args.gradient_accumulation_steps
    print('Print freq:', print_freq, "Eval freq:", eval_freq)

    for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
        tr_loss = 0
        nb_tr_steps = 0
        with tqdm(total=len(train_dataloader)//args.gradient_accumulation_steps) as bar:
            for step, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch, labels_batch = batch
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    loss = model(text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch, labels_batch)

                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    nb_tr_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if nb_tr_steps and nb_tr_steps % print_freq == 0:
                        bar.update(min(print_freq, nb_tr_steps))
                        time.sleep(0.02)
                        print(global_step, tr_loss / nb_tr_steps)
                        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))

                    if global_step and global_step % eval_freq == 0:
                        val_result = eval_running_model(val_dataloader)
                        print('Global Step %d VAL res:\n' % global_step, val_result)
                        log_wf.write('Global Step %d VAL res:\n' % global_step)
                        log_wf.write(str(val_result) + '\n')

                        if val_result['em'] > best_eval_loss:
                            best_eval_loss = val_result['em']
                            # save model
                            print('[Saving at]', state_save_path)
                            log_wf.write('[Saving at] %s\n' % state_save_path)
                            torch.save(model.state_dict(), state_save_path)
                log_wf.flush()

        # add a eval step after each epoch
        val_result = eval_running_model(val_dataloader)
        print('Epoch %d, Global Step %d VAL res:\n' % (epoch, global_step), val_result)
        log_wf.write('Global Step %d VAL res:\n' % global_step)
        log_wf.write(str(val_result) + '\n')

        if val_result['em'] > best_eval_loss:
            best_eval_loss = val_result['em']
            # save model
            print('[Saving at]', state_save_path)
            log_wf.write('[Saving at] %s\n' % state_save_path)
            torch.save(model.state_dict(), state_save_path)
        print(global_step, tr_loss / nb_tr_steps)
        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
