#!/bin/bash 
#SBATCH --partition=sm,titanx,2080
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH -x sls-2080-2
#SBATCH --open-mode=append
#SBATCH --mem=8000
#SBATCH --job-name="ext-code"
#SBATCH --output=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs/ext-code-%j.out

. /data/sls/u/wnhsu/code/davenet_vq/iclr20_oss/env.sh

set -exu

###########
# LibriTTS
###########

exp_root="/data/sls/temp/wnhsu/experiments/davenet_vq/iclr20/exps/PlacesEnglish400k"
model_path="$exp_root/batch1/ResDavenetVQ_c9_d2222_vqsize1024_vqon01100_jitter12_adam_warmstart_vq01000/models/best_audio_model.pth"
layer='quant3'

src_files=""
src_files="$src_files ./filelists/original/libritts_train460_train.txt"
src_files="$src_files ./filelists/original/libritts_train460_valid.txt"
src_files="$src_files ./filelists/original/libritts_train460_valid100.txt"

filelist_root="./filelists/rdvq_01000_01100"
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
