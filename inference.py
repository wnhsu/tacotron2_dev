#!/usr/bin/env python
# coding: utf-8

"""
E.g.

python inference.py \
    --ckpt_path=./exps/finished/basic_ljs_run1/checkpoint_77000 \
    --ckpt_args_path=./exps/finished/basic_ljs_run1/args.json \
    --out_dir=./exps/eval/basic_ljs_run1_step77000
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
import torch
from tqdm import tqdm

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence, code_to_sequence, SOS_TOK, EOS_TOK
from denoiser import Denoiser
from utils import load_code_dict, load_filepaths_and_text


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str,
                    default='./exps/finished/ljs_code_mdl-01000-01100_vq3_chunk_run1/checkpoint_77000')
parser.add_argument('--ckpt_args_path', type=str,
                    default='./exps/finished/ljs_code_mdl-01000-01100_vq3_chunk_run1/args.json')
parser.add_argument('--waveglow_path', type=str,
                    default='./pretrained/waveglow_256channels_new.pt')
parser.add_argument('--filelist', type=str,
                    default='filelists/rdvq_01000_01100/ljs_audio_text_test_filelist.txt')
parser.add_argument('--filelist_format', type=str, default='filelist', choices=['filelist', 'textonly'],
                    help='textonly is | separated textid and text')
parser.add_argument('--out_dir', type=str, default='./exps/eval/tmp')
parser.add_argument('--num_utts', type=int, default=5)
parser.add_argument('--label', type=int, default=None)
parser.add_argument('--max_decoder_steps', type=int, default=-1,
                    help='if not -1, overwrite the argument to synthesize longer utterances')
parser.add_argument('--append_str', type=str, default='',
                    help='append this to the end of each text')
parser.add_argument('--dump_att', action='store_true', dest='dump_att',
                    help='dump attention matrix if True')
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
if args.filelist_format == 'filelist':
    data = load_filepaths_and_text(args.filelist)
else:
    with open(args.filelist) as f:
        data = [line.rstrip().split('|', 1) for line in f]
code_dict = load_code_dict(hparams.code_dict, hparams.add_sos, hparams.add_eos)


# Helper functions
def get_text(idx):
    if args.filelist_format == 'filelist':
        return data[idx]['text']
    return data[idx][1]

def get_inp_str(idx):
    if args.filelist_format == 'textonly':
        return data[idx][1]
    elif hparams.text_or_code == 'text':
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

def get_textid(idx):
    if args.filelist_format == 'filelist':
        return str(idx)
    else:
        return data[idx][0]

def synthesize_one(idx, label=None):
    text = '(lab=%s) %s' % (label, get_text(idx))
    sequence = get_inp_ids(idx)
    if label is not None:
        label = torch.LongTensor(1).cuda().fill_(label)

    (mel_outputs, mel_outputs_postnet, _, alignments,
     has_eos) = model.inference(sequence, label, ret_has_eos=True)

    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
    plt.suptitle(text if len(text) < 150 else (text[:150] + '...'))
    plt.savefig('%s/spec_and_att_%s.png' % (args.out_dir, get_textid(idx)))
    plt.close()

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    sf.write('%s/wg_%s.wav' % (args.out_dir, get_textid(idx)),
             audio[0].data.cpu().float().numpy(), hparams.sampling_rate)

    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    # librosa write_wav is problematic, which uses 32-bit floating point
    # encoding that are not supported by the firefox browser
    sf.write('%s/wg_dn_%s.wav' % (args.out_dir, get_textid(idx)),
             audio_denoised[0].data.cpu().numpy(), hparams.sampling_rate)

    if args.dump_att:
        path = '%s/aud_att_%s.npy' % (args.out_dir, get_textid(idx))
        with open(path, 'wb') as f:
            # remove sos and eos frame in the alignments
            if hparams.add_sos:
                alignments = alignments[:, :, 1:]
            if hparams.add_eos:
                alignments = alignments[:, :, :-1]
            np.save(f, alignments[0].detach().float().cpu().numpy())
            print(alignments[0].detach().float().cpu().numpy().shape)

    return len(sequence[0]), has_eos

# Run
os.makedirs(args.out_dir, exist_ok=True)
num_utts = len(data)
if args.num_utts > 0:
    num_utts = min(args.num_utts, len(data))

has_eos_lst = []
for idx in tqdm(range(num_utts), total=num_utts):
    seqlen, has_eos = synthesize_one(idx, args.label)
    has_eos_lst.append(int(has_eos))
    # print('synthesizing input %d / %d (seqlen=%d)' % (idx+1, num_utts, seqlen))

with open('%s/has_eos.txt' % args.out_dir, 'w') as f:
    for has_eos in has_eos_lst:
        f.write('%s\n' % has_eos)
