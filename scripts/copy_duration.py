import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('inp', type=str, help='input filelist of the original format')
parser.add_argument('ref', type=str, help='reference path with duration info')
parser.add_argument('out', type=str, help='output path')
args = parser.parse_args()

with open(args.inp, 'r') as f:
    inp_data = [json.loads(line.rstrip()) for line in f]
with open(args.ref, 'r') as f:
    ref_data = [json.loads(line.rstrip()) for line in f]
    ref_audio2dur = {d['audio']: d['duration'] for d in ref_data}
    del ref_data

with open(args.out, 'w') as f:
    for d in inp_data:
        d['duration'] = ref_audio2dur[d['audio']]
        f.write(json.dumps(d, sort_keys=True) + '\n')
