import json
import argparse
import random
random.seed(0) # for reproducibility

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
    # handle a formatting bug in the data
    try:
        option = options[0]
    except:
        option = options

    correct_id, correct_text = option['candidate-id'], '{} {}'.format(option['speaker'], option['utterance']).replace('\t', ' ')
    context = ['{} {}'.format(message['speaker'], message['utterance']).replace('\t', ' ') for message in messages]
    outfile.write('{}\t{}\t{}\n'.format(1, '\t'.join(context), correct_text))

    # write negative samples
    negs = content['options-for-next']
    if args.mode != 'train': # for evaluation, pos:neg = 1:99
        cnt = 0
        for neg in negs:
            if neg['candidate-id'] != correct_id:
                cnt += 1
                neg_text = option['speaker'] + ' ' + neg['utterance']
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg_text))
        # handle a bug in the data where some instances do not have 99 negatives...
        if cnt != 99:
            diff = 99 - cnt
            for _ in range(diff):
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg_text))
    else: # for training, pos:neg = 1:1
        neg_indices = list(range(len(negs)))
        while True:
            neg_idx = random.choice(neg_indices)
            # handle a bug in the data where some instances have the correct id in negatives...
            if negs[neg_idx]['candidate-id'] != correct_id:
                neg_text = '{} {}'.format(option['speaker'], negs[neg_idx]['utterance'])
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg_text))
                break
