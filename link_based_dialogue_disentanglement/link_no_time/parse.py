import os
from collections import defaultdict
import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str)
parser.add_argument("--num_contexts", type=int)
parser.add_argument("--mode", type=str)
args = parser.parse_args()

res = []
parsed = set()

outfile = open('{}.txt'.format(args.mode), 'w')

for filename in tqdm.tqdm(os.listdir(args.dir_path)):
    date = filename.split('.')[0]
    if date in parsed:
        continue
    parsed.add(date)

    # read annotation file
    annotations = open(os.path.join(args.dir_path, date+'.annotation.txt')).read().splitlines()
    connections = []
    for annotation in annotations:
        first, second = annotation[:-2].split()
        connections.append([int(first), int(second)])
    
    # read text file
    text = open(os.path.join(args.dir_path, date+'.ascii.txt')).read().splitlines()
    
    max_context = args.num_contexts
    # we assume a message only relates to at most max_context sentences preceding it
    # sliding window
    slide = 1
    # starting from 1000
    for i in range(1000, len(text), slide): # 0-based
        childbyparent = {} # key/child is the later sentence, value/parent is the preceding sentence
        # however, some pair may have values outside the above range
        # we simply discard this kind of pairs
        # inclusive
        min_id = i
        max_id = min(i + max_context, len(text)) - 1
        if max_id - min_id + 1 < max_context:
            continue
        cur_connections = []
        for connection in connections:
            if min_id <= connection[0] <= max_id and min_id <= connection[1] <= max_id and connection[0] != connection[1]:
                cur_connections.append(connection)

        for j in range(min_id, max_id+1):
            tokens = text[j].split()
            if tokens[0][0] == '[' and tokens[0][-1] == ']': # time, skip
                tokens = tokens[1:]
                if tokens[0][0] == '<' and tokens[0][-1] == '>': # reduce seq length
                    tokens[0] = tokens[0][1:-1]
                outfile.write('{}\n'.format(' '.join(tokens)))
            else:
                outfile.write('{}\n'.format(text[j]))
            

        for cc in cur_connections:
            # we always pick shorter and later links
            if cc[1] in childbyparent:
                if cc[0] > childbyparent[cc[1]]:
                    childbyparent[cc[1]] = cc[0]
            else:
                childbyparent[cc[1]] = cc[0]
        # now, some of the children do not have parents
        # their parents should be the dummy root
        # find the intersection
        id_set = set(list(range(min_id, max_id+1)))
        diff = id_set - set(list(childbyparent.keys()))
        for d in diff:
            childbyparent[d] = float('-inf')
        # offset, 0 is the dummy root
        childbyparent = {k-min_id+1: v-min_id+1 for k, v in childbyparent.items()}
        # add dummy root
        for k, v in childbyparent.items():
            if v == float('-inf'):
                childbyparent[k] = 0
        # add -1 just for representation
        # same as turbo parser representation
        links = [str(-1)]
        for j in range(1, len(childbyparent)+1):
            links.append(str(childbyparent[j]))
        outfile.write('{}\n'.format(' '.join(links)))
