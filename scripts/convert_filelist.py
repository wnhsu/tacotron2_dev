import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('inp', type=str, help='input filelist of the original format')
parser.add_argument('out', type=str, help='output path')
args = parser.parse_args()

with open(args.inp, 'r') as fin, open(args.out, 'w') as fout:
    for line in fin:
        audio, text = line.rstrip().split('|')
        entry = {'audio': audio, 'text': text}
        fout.write(json.dumps(entry, sort_keys=True) + '\n')
