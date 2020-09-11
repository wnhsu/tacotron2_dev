#!/bin/usr/python

import argparse
import json
import librosa
import time
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = [json.loads(line.rstrip()) for line in f]

    def __getitem__(self, idx):
        return librosa.get_duration(filename=self.data[idx]['audio']), idx

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inp', type=str, help='input filelist of the original format')
    parser.add_argument('out', type=str, help='output path')
    args = parser.parse_args()

    start = time.time()
    loader = DataLoader(AudioDataset(args.inp), batch_size=100, 
                        shuffle=False, num_workers=32, drop_last=False)
    print('Start processing %s, %d utterances in total' % (
          args.inp, len(loader.dataset)))
    with open(args.out, 'w') as f:
        nutts = 0
        for durs, indices in loader:
            for dur, idx in zip(durs, indices):
                datum = dict(loader.dataset.data[idx])
                datum['duration'] = dur.cpu().item()
                f.write(json.dumps(datum, sort_keys=True) + '\n')
                nutts += 1
                
                if nutts % 1000 == 0:
                    print('Processing %d utts takes %.f(s)' % (
                          nutts, time.time() - start))
    
    print('Finished. Processing %s takes %.f(s)' % (args.inp, time.time() - start))
