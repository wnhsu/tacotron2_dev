#!/bin/bash 

. env.sh

set -eu

############
# Flickr8k #
############

tar_dir="/data/sls/temp/wnhsu/data/cv/flickr8k/mel"
for s in tr dt tt; do
  python scripts/make_audio_spec.py \
    filelists/original/flickr8k_${s}.txt \
    ${tar_dir}/data ${tar_dir}/spec_${s}.txt \
    --nlevels=2
done

for s in tr dt tt; do
  python scripts/dump_mel.py \
    --spec_path ${tar_dir}/spec_${s}.txt
done


##############
# SpeechCOCO #
##############

tar_dir="/data/sls/temp/wnhsu/data/cv/speech_coco/mel"
for s in tr dt tt; do
  python scripts/make_audio_spec.py \
    filelists/original/speechcoco_${s}.txt \
    ${tar_dir}/data ${tar_dir}/spec_${s}.txt \
    --nlevels=1
done


