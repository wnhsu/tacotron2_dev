#!/bin/bash 
#SBATCH --partition=sm,titanx,2080
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH -x sls-2080-2
#SBATCH --open-mode=append
#SBATCH --mem=80000
#SBATCH --job-name="ext-code"
#SBATCH --output=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs/ext-code-%j.out

. /data/sls/u/wnhsu/code/davenet_vq/iclr20_oss/env.sh

set -exu

###########
# SpokenCoco
###########

# src_dir="/data/sls/temp/wnhsu/data/cv/spoken_coco/filelist"
# python ./scripts/add_duration.py "${src_dir}/spokencoco_dt.txt" "./filelists/original/spokencoco_dt.txt"
# python ./scripts/add_duration.py "${src_dir}/spokencoco_tt.txt" "./filelists/original/spokencoco_tt.txt"
# python ./scripts/add_duration.py "${src_dir}/spokencoco_tr.txt" "./filelists/original/spokencoco_tr.txt"

exp_root="/data/sls/temp/wnhsu/experiments/davenet_vq/iclr20_oss/exps/PlacesEnglish400k_oss_batch2"
model_path="$exp_root/finished/RDVQ_2gpu_vqon01110_bs128_std1_nonnegF_seed-01000-01100/models/best_audio_model.pth"
layer='quant3'

src_files=""
src_files="$src_files ./filelists/original/spokencoco_dt.txt"
src_files="$src_files ./filelists/original/spokencoco_tt.txt"
src_files="$src_files ./filelists/original/spokencoco_tr.txt"

filelist_root="./filelists/rdvq_01000_01100_01110"
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
    --code_embedding_path=$filelist_root/code_embedding_$layer \
    --batch_size=50
done


