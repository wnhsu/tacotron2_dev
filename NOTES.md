## Installation Notes
* change `numpy==1.13.3` to `numpy`. version 1.13.3 is too old for other packages
* change `tensorboardX==1.1` to `tensorboardX` for similar reasons
* change `tensorflow` to `tensorflow-gpu==1.14`. tf.contrib.training.HParams is
  not in tf2.0

## Code Adaptation Notes
* make `get_mask_from_lengths()` return boolean instead of byte mask, because
  `masked_fill_` does not accept byte mask anymore.
* pass `dataformats` to `add_image()` in `Tacotron2Logger.log_validation()`.

## Pretrained Model Notes
* The commit number for `waveglow` is out-dated. Need to manually run `git
  submodule update --remote`
* Need to convert the model with `python waveglow/convert_model.py <old> <new>`

## Run commands

Single-GPU
```
srun --gres=gpu:1 -p sm,2080,titanx \
  python train.py -o <expdir> -l log \
  --hparams=batch_size=40,fp16_run=True
```

Multi-GPU
```
srun --gres=gpu:2 -p sm,2080,titanx \
  python -m multiproc train.py -o <expdir> -l log \
  --hparams=batch_size=40,fp16_run=True,distributed_run=True
```

## TODOs

### Features to add
* Save audio samples at each checkpoint
* change filelist to be list of json. Each entry has additional fields
  including speaker-id, duration, text, raw code (uncollapsed).
* Initialize code/text embedding
* pass speaker ID as input to the model
* infer global voice representations with an audio encoder
* AE/VAE/GST/GMVAE formulation for the objective
* data proprocessing
  * random audio/code chunking
  * silence removal

### Experiments
1. Train on LJSpeech with the original code
2. Add speaker 
