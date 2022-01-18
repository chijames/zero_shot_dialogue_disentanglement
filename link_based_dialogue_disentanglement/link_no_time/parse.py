import os
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str)
parser.add_argument("--num_contexts", type=int)
parser.add_argument("--mode", type=str)
args = parser.parse_args()

outfile = open('{}.txt'.format(args.mode), 'w')
outids = open('{}_ids.txt'.format(args.mode), 'w')
parsed = set()

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
    
    # we assume a message only relates to at most args.num_contexts sentences preceding it
    # the first 1000 sentences (id zero-based) are context without link annotations
    for min_id in range(1000-args.num_contexts+1, len(text)-args.num_contexts+1):
        max_id = min_id + args.num_contexts
        cur_connections = []
        for connection in connections:
            # some sentence pairs may have reside outside the [min_id, max_id) range, and we simply discard this kind of pairs
            # we also discard self link, which will be taken care of later
            if min_id <= connection[0] < max_id and min_id <= connection[1] < max_id and connection[0] != connection[1]:
                cur_connections.append(connection)

        # write the text part
        for i in range(min_id, max_id):
            tokens = text[i].split()
            if tokens[0][0] == '[' and tokens[0][-1] == ']': # skip time to be consistent with the entangled response selection training data
                tokens = tokens[1:]
                if tokens[0][0] == '<' and tokens[0][-1] == '>': # a trick to reduce seq length, not important
                    tokens[0] = tokens[0][1:-1]
                outfile.write('{}\n'.format(' '.join(tokens)))
            else:
                outfile.write('{}\n'.format(text[i]))

        child2parent = {} # key/child is the later sentence, value/parent is the preceding sentence that the child points to
        for cc in cur_connections:
            # we always pick the most recent (with a larger id) links
            child2parent[cc[1]] = max(cc[0], child2parent[cc[1]]) if cc[1] in child2parent else cc[0]

        # some of the children/sentences do not have parents, and their parents should be the dummy root
        orphans = set(list(range(min_id, max_id))) - set(list(child2parent.keys()))
        # we make the ids 0-indexed first (-min_id) and then shift all ids by 1 (+1) to accommodate the dummy root (id=0)
        child2parent = {k-min_id+1: v-min_id+1 for k, v in child2parent.items()}
        for orphan in orphans:
            # apply the same 0-indexed and shift by 1 logic
            child2parent[orphan-min_id+1] = 0

        assert len(child2parent) == args.num_contexts
        # add -1 just for the consistency with turbo parser
        # only the last numbers are used in this work
        # you can safely ignore all the other numbers
        links = [str(-1)]
        for j in range(1, args.num_contexts+1):
            links.append(str(child2parent[j]))
        outfile.write('{}\n'.format(' '.join(links)))
        outids.write('{} {}\n'.format(date, max_id-1)) # note that the range is [min_id, max_id), so we need to -1 to compensate for that
