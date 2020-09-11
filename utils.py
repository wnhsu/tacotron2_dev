import io
import json
import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.io.wavfile import read
from text import SOS_TOK, EOS_TOK


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


def read_binary_audio(bin_data, tar_sr=None):
    """
    read binary audio (`bytes` or `uint8` `numpy.ndarray`) to `float32`
    `numpy.ndarray`

    RETURNS:
        data (np.ndarray) : audio of shape (n,) or (2, n)
        tar_sr (int) : sample rate
    """
    data, ori_sr = sf.read(io.BytesIO(bin_data), dtype='float32')
    data = data.T
    if (tar_sr is not None) and (ori_sr != tar_sr):
        data = librosa.resample(data, ori_sr, tar_sr)
    else:
        tar_sr = ori_sr
    data = np.clip(data, -1, 1)
    data = data * 32768.0
    return torch.FloatTensor(data.astype(np.float32)), tar_sr


def load_filepaths_and_text(filename):
    with open(filename, encoding='utf-8') as f:
        data = [json.loads(line.rstrip()) for line in f]
    return data


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def load_code_dict(path, add_sos=False, add_eos=False):
    if not path:
        return {}

    with open(path, 'r') as f:
        codes = ['_'] + [line.rstrip() for line in f]  # '_' for pad
    code_dict = {c: i for i, c in enumerate(codes)}

    if add_sos:
        code_dict[SOS_TOK] = len(code_dict)
    if add_eos:
        code_dict[EOS_TOK] = len(code_dict)
    assert(set(code_dict.values()) == set(range(len(code_dict))))

    return code_dict


def load_obs_label_dict(path):
    if not path:
        return {}
    with open(path, 'r') as f:
        obs_labels = [line.rstrip() for line in f]
    return {c: i for i, c in enumerate(obs_labels)}
