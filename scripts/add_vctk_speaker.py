import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('inp', type=str, help='input filelist of the original format')
parser.add_argument('out', type=str, help='output path')
args = parser.parse_args()

with open(args.inp, 'r') as f:
    data = [json.loads(line.rstrip()) for line in f]

with open(args.out, 'w') as f:
    for datum in data:
        uttid = os.path.splitext(os.path.basename(datum['audio']))[0]
        datum['speaker'] = uttid.split('_')[0]
        f.write(json.dumps(datum, sort_keys=True) + '\n')
