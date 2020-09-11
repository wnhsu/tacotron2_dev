#!/bin/bash 

. env.sh

set -exu

out_root="/data/sls/temp/wnhsu/data/asr/places_eng_400k/ljs_code_mdl-01000-01100_vq3_chunk_run1"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/places_eng_400k_val_filelist.txt" \
  --out_dir="${out_root}/valid" --num_utts=-1 --append_str=" 263 208"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/places_eng_400k_train_filelist.txt" \
  --out_dir="${out_root}/train" --num_utts=-1 --append_str=" 263 208"

