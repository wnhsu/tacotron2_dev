#!/bin/bash 

. env.sh

set -exu

out_root="/scratch/wnhsu/data/asr/spokencoco/tmp"
# out_root="/scratch/wnhsu/data/asr/spokencoco_noend/ljs_code_mdl-01000-01100_vq3_chunk_run1"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/spokencoco_tt.txt" \
  --out_dir="${out_root}/test" --num_utts=-1 --append_str=" 263 208"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/spokencoco_dt.txt" \
  --out_dir="${out_root}/valid" --num_utts=-1 --append_str=" 263 208"

python dump_asr_dataset.py \
  --filelist="./filelists/rdvq_01000_01100/spokencoco_tr.txt" \
  --out_dir="${out_root}/train" --num_utts=-1 --append_str=" 263 208"

tar -zcvf /data/sls/temp/wnhsu/data/cv/spokencoco_syn_asr.tar.gz $out_root && rm -rf $out_root
