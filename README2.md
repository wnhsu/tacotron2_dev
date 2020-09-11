# VAE-Tacotron 2

## Key modifications:
- Take JSON list file as input for more flexible data specifications
- Support length filtering to avoid OOM issues
- Support loading audio from `*.wav` or from `*.hdf5`
- Choices of text or code as TTS input
- Implement label conditioning (e.g., speaker)
- Implement stochastic latent representation encoder


## Related directories
`/usr/users/wnhsu/code/tacotron2_factory/tacotron2_hdf5_loader/dump_hdf5_dataset.py`:
Generate HDF5 dataset for faster audio loading

`/usr/users/wnhsu/code/davenet_vq/iclr20_oss/extract_tts_features.py`: Map
audio to code sequence for code-to-speech model


## Main scripts:

### Preprocess
`./scripts/preprocess/convert_filelist.py`: convert the original filelist (lines of
`wav_path transcript`) to JSON list file.

`./scripts/preprocess/add_duration.py`: add a `duration` field to a filelist, used for
length filtering in TTS model training.

`./scripts/preprocess/copy_duration.py`: copy the `duration` field

`./scripts/preprocess/dump_attributes.py`: dump all values of a given attribute
(e.g., speaker) to a file (e.g., speaker list).

`./scripts/ext_rdvq_codes`: load a trained ResDavenetVQ model to extract code
sequences at the specified layer for audio files in the given filelist.

### Train
`./train.py`: train a TTS model

### Evaluation
`./inference.py`: synthesize utterances with default label/representation.
Accept JSON list file or one-transcript-per-line file as input.

`./inference.ipynb`:

`./inference_attr.ipynb`:

### Utilities
`./dump_asr_dataset.py`: generate an ASR dataset of synthesize audios for kaldi.

`./scripts/dump_mel/dump_mel.py`: pre-compute Mel features (unused).


## Recipe scripts:

`./scripts/run.sh`: run training
`./scripts/run_eval.sh`: run decoding
