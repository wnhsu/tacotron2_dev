import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text, load_code_dict, load_obs_label_dict
from text import text_to_sequence, code_to_sequence, sample_code_chunk


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.data = load_filepaths_and_text(audiopaths_and_text)
        self.text_or_code = hparams.text_or_code
        self.text_cleaners = hparams.text_cleaners
        self.code_key = hparams.code_key
        self.code_dict = load_code_dict(hparams.code_dict)
        self.collapse_code = hparams.collapse_code
        self.chunk_code = hparams.chunk_code
        self.obs_label_dict = load_obs_label_dict(hparams.obs_label_dict)
        self.obs_label_key = hparams.obs_label_key
        self.chunk_size = -1
        self.min_chunk_size = 1
        self.always_chunk = False
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.max_wav_nframe = -1
        if hparams.max_wav_len > 0:
            spf = hparams.hop_length / hparams.sampling_rate
            self.max_wav_nframe = int(hparams.max_wav_len / spf)
        random.seed(1234)
        random.shuffle(self.data)

    def set_chunking_mode(self, always_chunk, min_chunk_size=None):
        print('Changing always_chunk from %s to %s' % (
              self.always_chunk, always_chunk))
        self.always_chunk = always_chunk
        if always_chunk and min_chunk_size is not None:
            assert(min_chunk_size <= self.chunk_size)
            print('Changing min_chunk_size from %d to %d' % (
                  self.min_chunk_size, min_chunk_size))
            self.min_chunk_size = min_chunk_size

    def set_code_chunk_size(self, chunk_size):
        assert(self.text_or_code == 'code')
        assert(chunk_size > 0)
        print('Changing max code-chunk size from %d to %d' % (
              self.chunk_size, chunk_size))
        self.chunk_size = chunk_size

    def get_mel_symbol_pair(self, datum):
        if self.text_or_code == 'text':
            symbol = self.get_text(datum['text'])
        elif self.text_or_code == 'code':
            code = datum[self.code_key].split()
            if self.chunk_code:
                tot = len(code)
                if self.chunk_size == -1:
                    chunk_size = tot
                else:
                    chunk_size = min(self.chunk_size, tot)

                if self.always_chunk:
                    min_chunk_size = min(self.min_chunk_size, chunk_size)
                    chunk_size = np.random.randint(min_chunk_size,
                                                   chunk_size + 1)
                code, start, end = sample_code_chunk(code, chunk_size)
            symbol = self.get_code(code)
        else:
            raise ValueError('%s not supported' % self.text_or_code)

        mel = self.get_mel(datum['audio'])
        if self.text_or_code == 'code' and self.chunk_code:
            fpc = float(mel.size(1)) / tot
            fstart = int(np.floor(start * fpc))
            fend = int(np.ceil(end * fpc))
            assert(fstart != fend)
            mel = mel[:, fstart:fend]

        return (symbol, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        if self.max_wav_nframe > 0:
            melspec = melspec[:, :self.max_wav_nframe]

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_code(self, code):
        code_norm = torch.IntTensor(
                code_to_sequence(code, self.code_dict, self.collapse_code))
        return code_norm

    def __getitem__(self, index):
        """
        return dummy obs_label_id (0) if obs_label_key is not set.
        """
        symbol, mel = self.get_mel_symbol_pair(self.data[index])
        obs_label_id = torch.tensor(0).int()
        if self.obs_label_key:
            obs_label = self.data[index][self.obs_label_key]
            obs_label_id = self.obs_label_dict[obs_label]
            obs_label_id = torch.tensor(obs_label_id).int()
        return symbol, mel, obs_label_id

    def __len__(self):
        return len(self.data)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # re-order observed labels
        obs_labels = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            obs_labels[i] = batch[ids_sorted_decreasing[i]][2]

        return (text_padded, input_lengths, obs_labels, 
                mel_padded, gate_padded, output_lengths)
