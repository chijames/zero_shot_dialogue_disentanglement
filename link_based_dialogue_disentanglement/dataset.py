import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import pickle


class SelectionDataset(Dataset):
    def __init__(self, file_path, args, tokenizer, sample_cnt=None):
        self.max_contexts_length = args.max_contexts_length
        self.data_source = []
        self.tokenizer = tokenizer
        with open(file_path, encoding='utf-8') as f:
            i = 0
            group = {
                'context': [],
                'labels': []
            }
            for line in f:
                line = line.strip()
                if (i+1)%(args.max_num_contexts+1) != 0: # text
                    group['context'].append(line)
                else:
                    ans = int(line.split(' ')[-1]) - 1
                    temp = [0]*args.max_num_contexts
                    temp[ans] = 1
                    group['labels'].append(temp)

                    self.data_source.append(group)
                    group = {
                        'context': [],
                        'labels': []
                    }
                i += 1
        if sample_cnt is not None:
            random.shuffle(self.data_source)
            subset = int(len(self.data_source)*sample_cnt)
            self.data_source = self.data_source[:subset]

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        return self.data_source[index]

    def batchify_join_str(self, batch):
        text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch, labels_batch = [], [], [], []
        for sample in batch:
            contexts, response, labels = sample['context'], sample['context'][-1], sample['labels'][0]
            tokenized_dict = self.tokenizer(contexts, [response]*len(contexts), padding='max_length', truncation=True, max_length=self.max_contexts_length*2)
            text_token_ids_list, text_input_masks_list, type_ids_list = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']

            text_token_ids_list_batch.append(text_token_ids_list)
            text_input_masks_list_batch.append(text_input_masks_list)
            type_ids_batch.append(type_ids_list)
            labels_batch.append(labels)

        long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch]
        text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch = (
            torch.tensor(t, dtype=torch.long) for t in long_tensors)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return text_token_ids_list_batch, text_input_masks_list_batch, type_ids_batch, labels_batch
