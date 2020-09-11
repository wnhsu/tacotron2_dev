import sys
sys.path.insert(0, '/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017_dev')
import argparse
import numpy as np
import os
import time
import torch
from hparams import create_hparams
from layers import TacotronSTFT
from utils import load_wav_to_torch

def get_mel(stft, path, max_wav_value=32768.0):
    audio, sampling_rate = load_wav_to_torch(path, stft.sampling_rate)
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec.detach().cpu().numpy()

def load_audio_spec(spec_path):
    """
    Each line in spec_path is `src_path tar_path`
    """
    with open(spec_path, 'r') as f:
        paths_list = [line.rstrip().split() for line in f]
    
    for paths in paths_list:
        if len(paths) != 2:
            raise ValueError('File is ill-formatted. found %s' % (paths,))
    return paths_list

def main(hparams, spec_path, overwrite=False):
    paths_list = load_audio_spec(spec_path)
    print('Loading audio inp/out paths from %s' % spec_path)

    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)
    
    print('Total %d audio files to extract features.' % len(paths_list))
    start = time.time()
    for i, (src_path, tar_path) in enumerate(paths_list):
        if os.path.exists(tar_path) and not overwrite:
            pass
        mel = get_mel(stft, src_path, hparams.max_wav_value)
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        np.save(tar_path, mel)
        if (i + 1) % 100 == 0:
            print('...processed %d audios' % (i + 1))
    print('Finished in %.fs' % (time.time() - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_path', type=str, required=True)
    parser.add_argument('--hparams', type=str, default='', required=False)

    args = parser.parse_args()
    hparams = create_hparams(args.hparams, verbose=True)

    main(hparams, args.spec_path)
