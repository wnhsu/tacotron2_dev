#!/bin/bash 

set -exu 

src_dir="./filelists/original"
tar_dir="./filelists/original"

paths=""
paths="$paths ${src_dir}/ljs_audio_text_test_filelist.txt"
paths="$paths ${src_dir}/ljs_audio_text_val_filelist.txt"
paths="$paths ${src_dir}/ljs_audio_text_train_filelist.txt"
paths="$paths ${src_dir}/vctk_trimmed_train_filelist.txt"
paths="$paths ${src_dir}/vctk_trimmed_valid_filelist.txt"
paths="$paths ${src_dir}/libritts_train460_train.txt"
paths="$paths ${src_dir}/libritts_train460_valid.txt"
paths="$paths ${src_dir}/libritts_train460_valid100.txt"
paths="$paths ${src_dir}/libritts_train960_train.txt"
paths="$paths ${src_dir}/libritts_train960_valid.txt"
paths="$paths ${src_dir}/libritts_train960_valid100.txt"
paths="$paths ${src_dir}/places_eng_400k_val_filelist.txt"
paths="$paths ${src_dir}/places_eng_400k_train_filelist.txt"

for p in $paths; do 
  python ./scripts/add_duration.py $p ${tar_dir}/$(basename $p)
done

#####
src_dir="./filelists/rdvq_01000_01100"
tar_dir="./filelists/rdvq_01000_01100"
basenames="flickr8k_tr.txt flickr8k_dt.txt flickr8k_tt.txt"
for f in $basenames; do
  python scripts/add_duration.py $src_dir/$f $tar_dir/$f 
done
