import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import pickle


class SelectionDataset(Dataset):
    def __init__(self, file_path, concat_transform, sample_cnt=None):
        self.concat_transform = concat_transform
        self.data_source = []
        with open(file_path, encoding='utf-8') as f:
            group = {
                'context': None,
                'responses': [],
                'labels': []
            }
            for line in f:
                split = line.strip().split('\t')
                lbl, context, response = int(split[0]), split[1:-1], split[-1]
                if lbl == 1 and len(group['responses']) > 0:
                    self.data_source.append(group)
                    group = {
                        'context': None,
                        'responses': [],
                        'labels': []
                    }
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
                group['responses'].append(response)
                group['labels'].append(lbl)
                group['context'] = context
            if len(group['responses']) > 0:
                self.data_source.append(group)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        group = self.data_source[index]
        context, responses, labels = group['context'], group['responses'], group['labels']
        transformed_text = self.concat_transform(context, responses)
        ret = transformed_text, labels

        return ret

    def batchify_join_str(self, batch):
        text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [], [], []
        labels_batch = []
        for sample in batch:
            text_token_ids_list, text_input_masks_list, text_segment_ids_list = sample[0]

            text_token_ids_list_batch.append(text_token_ids_list)
            text_input_masks_list_batch.append(text_input_masks_list)
            text_segment_ids_list_batch.append(text_segment_ids_list)
            labels_batch.append(sample[1])

        long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]

        text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (
            torch.tensor(t, dtype=torch.long) for t in long_tensors)

        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch
