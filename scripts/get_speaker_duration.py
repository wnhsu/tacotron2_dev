import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('filelist_path', type=str)
parser.add_argument('speaker_path', type=str)
args = parser.parse_args()

spk2dur = defaultdict(float)
with open(args.filelist_path) as f:
    for line in f:
        datum = json.loads(line.rstrip())
        spk2dur[datum['speaker']] += datum['duration']

spk2sid = dict()
with open(args.speaker_path) as f:
    for sid, line in enumerate(f):
        spk2sid[line.rstrip()] = sid

spk_sid_dur = [(spk, spk2sid[spk], spk2dur[spk]) for spk in spk2dur]
spk_sid_dur = sorted(spk_sid_dur, key=lambda x: -x[2])

print('%-20s %-20s %-20s' % ('SPEAKER', 'SID', 'DURATION (hr)'))
for spk, sid, dur in spk_sid_dur:
    print('%-20s %-20s %-20.2f' % (spk, sid, dur / 3600))
        
