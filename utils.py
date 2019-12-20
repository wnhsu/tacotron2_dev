import json
import librosa
import numpy as np
import torch
from scipy.io.wavfile import read


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path, sr=None):
    data, sr = librosa.load(full_path, sr=sr)
    data = np.clip(data, -1, 1)  # potentially out of [-1, 1] due to resampling
    data = data * 32768.0  # match values loaded by scipy
    return torch.FloatTensor(data.astype(np.float32)), sr


def load_filepaths_and_text(filename):
    with open(filename, encoding='utf-8') as f:
        data = [json.loads(line.rstrip()) for line in f]
    return data


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def load_code_dict(path):
    if not path:
        return {}
    with open(path, 'r') as f:
        codes = ['_'] + [line.rstrip() for line in f]  # '_' for pad
    return {c: i for i, c in enumerate(codes)}


def load_obs_label_dict(path):
    if not path:
        return {}
    with open(path, 'r') as f:
        obs_labels = [line.rstrip() for line in f]
    return {c: i for i, c in enumerate(obs_labels)}
