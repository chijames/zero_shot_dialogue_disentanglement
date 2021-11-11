import json
from collections import defaultdict
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
args = parser.parse_args()

with open('{}.json'.format(args.mode)) as infile:
    data = json.load(infile)

outfile = open('{}.txt'.format(args.mode), 'w')
for content in data:
    messages = content['messages-so-far']
    options = content['options-for-correct-answers']
    if len(options) == 0: # we skip no answers
        continue
    try:
        option = options[0]
    except:
        option = options
    correct_id = option['candidate-id']
    correct_text = option['speaker'] + ' ' + option['utterance'].strip()
    context = []
    for message in messages:
        text = message['speaker'] + ' ' + message['utterance']
        context.append(text.replace('\t', ' '))
    outfile.write('{}\t{}\t{}\n'.format(1, '\t'.join(context), correct_text))
    # write negative samples
    if args.mode != 'train':
        negs = content['options-for-next']
        cnt = 0
        for neg in negs:
            if neg['candidate-id'] != correct_id:
                cnt += 1
                neg_text = option['speaker'] + ' ' + neg['utterance'].strip()
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg_text))
        if cnt != 99:
            diff = 99 - cnt
            for _ in range(diff):
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg_text))
    if args.mode == 'train':
        negs = content['options-for-next']
        neg_indices = list(range(len(negs)))
        while True:
            neg_idx = random.choice(neg_indices)
            if negs[neg_idx]['candidate-id'] != correct_id:
                neg_text = option['speaker'] + ' ' + negs[neg_idx]['utterance'].strip()
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg_text))
                break
