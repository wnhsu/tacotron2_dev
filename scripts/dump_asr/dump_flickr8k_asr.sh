#!/bin/bash 

. env.sh

set -exu

out_root="/scratch/wnhsu/data/asr/flickr8k/ljs_code_mdl-01000-01100_vq3_chunk_run1"
# out_root="/data/sls/temp/wnhsu/data/asr/flickr8k/ljs_code_mdl-01000-01100_vq3_chunk_run1"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/flickr8k_tt.txt" \
  --out_dir="${out_root}/test" --num_utts=-1 --append_str=" 263 208"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/flickr8k_dt.txt" \
  --out_dir="${out_root}/valid" --num_utts=-1 --append_str=" 263 208"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/flickr8k_tr.txt" \
  --out_dir="${out_root}/train" --num_utts=-1 --append_str=" 263 208"

# tar -zcvf /data/sls/temp/wnhsu/data/cv/flickr8k_syn_asr.tar.gz $out_root && rm -rf $out_root
