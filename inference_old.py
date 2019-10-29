#!/usr/bin/env python
# coding: utf-8

"""
E.g.

python inference.py --ckpt_path=./exps/finished/basic_ljs_run1/checkpoint_77000 --ckpt_args_path=./exps/finished/basic_ljs_run1/args.json --out_dir=./exps/eval/basic_ljs_run1_step77000
"""

import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import sys
sys.path.append('waveglow/')
import librosa
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train_old import load_model  # for models prior to speaker embedding version
from text import text_to_sequence, code_to_sequence
from denoiser import Denoiser
from utils import load_code_dict, load_filepaths_and_text


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='./pretrained/tacotron2_statedict.pt')
parser.add_argument('--ckpt_args_path', type=str, default='')
parser.add_argument('--waveglow_path', type=str, default='./pretrained/waveglow_256channels_new.pt')
parser.add_argument('--filelist', type=str, default='filelists/rdvq_01000_01100/ljs_audio_text_test_filelist.txt')
parser.add_argument('--out_dir', type=str, default='./exps/eval')
parser.add_argument('--num_utts', type=int, default=5)
args = parser.parse_args()


hparams = create_hparams()
if args.ckpt_args_path:
    with open(args.ckpt_args_path, 'r') as f:
        args_dict = json.load(f)
    for k in args_dict:
        assert(hasattr(hparams, k))
        if getattr(hparams, k) != args_dict[k]:
            print('setting %20s from %20s to %20s' % (
                k, getattr(hparams, k), args_dict[k]))
            setattr(hparams, k, args_dict[k])
hparams.distributed_run = False


checkpoint_path = args.ckpt_path
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


waveglow_path = args.waveglow_path
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


data = load_filepaths_and_text(args.filelist)
code_dict = load_code_dict(hparams.code_dict)

def synthesize_one(idx):
    text = data[idx]['text']
    print('(TEXT LEN=%d) %s' % (len(text), text))
    
    if hparams.text_or_code == 'text':
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    else:
        code = data[idx][hparams.code_key].split()
        print('(CODE LEN=%d) %s' % (len(code), ' '.join(code)))
        sequence = np.array(code_to_sequence(code, code_dict, True))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    
    
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
    plt.suptitle(text if len(text) < 150 else (text[:150] + '...'))
    plt.savefig('%s/output_tacotron2_%d.png' % (args.out_dir, idx))
    
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    librosa.output.write_wav(
            '%s/output_waveglow_%d.wav' % (args.out_dir, idx),
            audio[0].data.cpu().float().numpy(), hparams.sampling_rate)
    
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    librosa.output.write_wav(
            '%s/output_waveglow_denoised_%d.wav' % (args.out_dir, idx),
            audio_denoised[0].data.cpu().numpy(), hparams.sampling_rate)

os.makedirs(args.out_dir, exist_ok=True)
for idx in range(args.num_utts):
    synthesize_one(idx)
