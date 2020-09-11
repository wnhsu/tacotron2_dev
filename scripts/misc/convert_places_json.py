import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('inp', type=str, help='input filelist of the original format')
parser.add_argument('out', type=str, help='output path')
args = parser.parse_args()

with open(args.inp, 'r') as f:
    data = json.load(f)['data']

with open(args.out, 'w') as f:
    for datum in data:
        datum['audio'] = datum.pop('wav')
        f.write(json.dumps(datum, sort_keys=True) + '\n')
