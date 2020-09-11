#!/usr/bin/env python
# coding: utf-8

"""
Adapted from inference.py, used to dump a dataset for training ASR with synthesized speech.
Output two files: wav.scp and text
E.g.

"""

import argparse
import librosa
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import os
import soundfile as sf
import sys
sys.path.append('waveglow/')
import time
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence, code_to_sequence, SOS_TOK, EOS_TOK
from denoiser import Denoiser
from utils import load_code_dict, load_filepaths_and_text


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str,
                    default='./exps/finished/ljs_code_mdl-01000-01100_vq3_chunk_run1/checkpoint_77000')
parser.add_argument('--ckpt_args_path', type=str,
                    default='./exps/finished/ljs_code_mdl-01000-01100_vq3_chunk_run1/args.json')
parser.add_argument('--waveglow_path', type=str,
                    default='./pretrained/waveglow_256channels_new.pt')
parser.add_argument('--filelist', type=str,
                    default='filelists/rdvq_01000_01100/ljs_audio_text_test_filelist.txt')
parser.add_argument('--out_dir', type=str, default='./exps/eval/tmp')
parser.add_argument('--num_utts', type=int, default=5)
parser.add_argument('--max_decoder_steps', type=int, default=-1,
                    help='if not -1, overwrite the argument to synthesize longer utterances')
parser.add_argument('--append_str', type=str, default='',
                    help='append this to the end of each text')
args = parser.parse_args()
print(args)


# Load hparams
hparams = create_hparams()
with open(args.ckpt_args_path, 'r') as f:
    args_dict = json.load(f)
for k in args_dict:
    assert(hasattr(hparams, k))
    if getattr(hparams, k) != args_dict[k]:
        print('setting %20s from %20s to %20s' % (
            k, getattr(hparams, k), args_dict[k]))
        setattr(hparams, k, args_dict[k])
hparams.distributed_run = False


# Load Tacotron / Waveglow / Denoiser
checkpoint_path = args.ckpt_path
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()
if args.max_decoder_steps != -1:
    model.decoder.max_decoder_steps = args.max_decoder_steps

waveglow_path = args.waveglow_path
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()

denoiser = Denoiser(waveglow)


# Load input sequences to synthesize
data = load_filepaths_and_text(args.filelist)
code_dict = load_code_dict(hparams.code_dict, hparams.add_sos, hparams.add_eos)


# Helper functions
def get_text(idx):
    return data[idx]['text']

def get_inp_str(idx):
    if hparams.text_or_code == 'text':
        return data[idx]['text']
    else:
        return data[idx][hparams.code_key]

def get_inp_ids(idx):
    inp_str = get_inp_str(idx)
    inp_str = inp_str + args.append_str
    if hparams.text_or_code == 'text':
        sequence = text_to_sequence(inp_str, ['english_cleaners'])
    else:
        codes = inp_str.split()
        if hparams.add_sos:
            codes = [SOS_TOK] + codes
        if hparams.add_eos:
            codes = codes + [EOS_TOK]
        sequence = code_to_sequence(codes, code_dict,
                                    collapse_code=True)
    return torch.from_numpy(np.array(sequence)[None, :]).cuda().long()

def get_uttid(idx):
    """if uttid does not exist, use index as uttid"""
    return data[idx].get('uttid', str(idx))


# Run
args.out_dir = os.path.abspath(args.out_dir)
num_utts = len(data)
if args.num_utts > 0:
    num_utts = min(args.num_utts, len(data))

os.makedirs('%s/wav' % args.out_dir, exist_ok=True)
with open('%s/meta.json'  % args.out_dir, 'w') as f:
    json.dump(vars(args), f, indent=2)
f_txt = open('%s/text' % args.out_dir, 'w')
f_wav = open('%s/wav.scp' % args.out_dir, 'w')
f_eos = open('%s/has_eos' % args.out_dir, 'w')
num_eos = 0

start = time.time()
with torch.no_grad():
    for idx in range(num_utts):
        uttid = get_uttid(idx)
        text = get_text(idx)
        sequence = get_inp_ids(idx)
        wav_path = '%s/wav/%s.wav' % (args.out_dir, uttid)
    
        _, mel_outputs_postnet, _, _ = model.inference(sequence)
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio_dn = denoiser(audio, strength=0.01)[0, 0].detach().cpu().numpy()
        eos = mel_outputs_postnet.size(-1) != model.decoder.max_decoder_steps

        sf.write(wav_path, audio_dn, hparams.sampling_rate)
        f_txt.write('%s %s\n' % (uttid, text.strip().replace('\n', ' ')))
        f_wav.write('%s %s\n' % (uttid, wav_path))
        f_eos.write('%s %s\n' % (uttid, int(eos)))
        num_eos += int(eos)
        
        if (idx + 1) % 50 == 0:
            print('...synthesized %d (%d has eos) / %d takes %.fs'
                  % (idx+1, num_eos, num_utts, time.time() - start))

f_txt.close()
f_wav.close()
