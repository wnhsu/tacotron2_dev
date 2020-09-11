#!/bin/bash 

set -exu

fs="ljs_audio_text_val_filelist.txt"
fs="$fs ljs_audio_text_test_filelist.txt"
fs="$fs ljs_audio_text_train_filelist.txt"

for f in $fs; do
  python ./scripts/copy_duration.py \
    ./filelists/rdvq_01000_01100/$f \
    ./filelists/original/$f \
    ./filelists/rdvq_01000_01100/$f
  
  python ./scripts/copy_duration.py \
    ./filelists/rdvq_conti_01000/$f \
    ./filelists/original/$f \
    ./filelists/rdvq_conti_01000/$f
done

fs="vctk_trimmed_train_filelist.txt"
fs="$fs vctk_trimmed_valid_filelist.txt"

for f in $fs; do
  python ./scripts/copy_duration.py \
    ./filelists/rdvq_01000_01100/$f \
    ./filelists/original/$f \
    ./filelists/rdvq_01000_01100/$f
done
