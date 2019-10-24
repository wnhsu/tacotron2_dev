#!/bin/bash 

. /data/sls/u/wnhsu/code/davenet_vq/iclr20_oss/env.sh

set -exu

# exp_root="/data/sls/temp/wnhsu/experiments/davenet_vq/iclr20/exps/PlacesEnglish400k"
# model_path="$exp_root/batch1/ResDavenetVQ_c9_d2222_vqsize1024_vqon01100_jitter12_adam_warmstart_vq01000/models/best_audio_model.pth"
exp_root="/data/sls/temp/wnhsu/experiments/davenet_vq/iclr20/exps/PlacesEnglish400k"
model_path="$exp_root/batch1/ResDavenetVQ_c9_d2222_vqsize1024_vqon01000_jitter12_adam_warmstart_conti/models/best_audio_model.pth"
layer='quant2'

src_files=""
src_files="$src_files ./filelists/ljs_audio_text_val_filelist.txt"
src_files="$src_files ./filelists/ljs_audio_text_test_filelist.txt"
src_files="$src_files ./filelists/ljs_audio_text_train_filelist.txt"
# src_files="$src_files ./filelists/rdvq_01000_01100_v2/ljs_audio_text_val_filelist.txt"
# src_files="$src_files ./filelists/rdvq_01000_01100_v2/ljs_audio_text_test_filelist.txt"
# src_files="$src_files ./filelists/rdvq_01000_01100_v2/ljs_audio_text_train_filelist.txt"

# filelist_root="./filelists/rdvq_01000_01100"
filelist_root="./filelists/rdvq_conti_01000"
mkdir -p $filelist_root 

code_key="code_$layer"

cat <<EOF >>$filelist_root/notes.txt
#####
audio model: $model_path
layer: $layer
code_key: $code_key
#####
EOF

for src_file in $src_files; do
  python /data/sls/u/wnhsu/code/davenet_vq/iclr20_oss/extract_tts_features.py \
    $model_path $src_file $filelist_root/$(basename $src_file) \
    --layer=$layer --code_key=$code_key \
    --code_dict_path=$filelist_root/code_dict_$layer \
    --code_embedding_path=$filelist_root/code_embedding_$layer
done
