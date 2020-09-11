#!/bin/bash 

step=77000
eval_dir="./exps/eval"

##################
# Text-to-Speech #
##################
# exp_dir="./exps/finished/basic_ljs_run1"
# exp_name=$(basename ${exp_dir})
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}

##################
# Code-to-Speech #
##################

# # Trained on WVQ code
# exp_dir="./exps/ljs_code_wavenetvq_run1/"
# exp_name=$(basename ${exp_dir})
# step=45000
# 
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/ljs \
#   --filelist="./filelists/wvq_places/ljs_audio_text_test_filelist.txt" \
#   --num_utts=20
# 
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/spokencoco \
#   --filelist="./filelists/wvq_places/spokencoco_tt.txt" \
#   --num_utts=20



# # Trained on LJS VQ2 (01000-01100)
# exp_dir="./exps/finished/ljs_code_mdl-01000-01100_vq2_run1"
# exp_name=$(basename ${exp_dir})
# 
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/ljs


# # Trained on LJS VQ3 (01000-01100)
# exp_dir="./exps/finished/ljs_code_mdl-01000-01100_vq3_run1"
# exp_name=$(basename ${exp_dir})
# 
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/ljs
# 
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/vctk \
#   --filelist="./filelists/rdvq_01000_01100/vctk_trimmed_valid_filelist.txt" \
#   --num_utts=20
#  
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/places \
#   --filelist="./filelists/rdvq_01000_01100/places_eng_400k_val_filelist.txt" \
#   --num_utts=20
#  
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/spokencoco \
#   --filelist="./filelists/rdvq_01000_01100/spokencoco_tt.txt" \
#   --num_utts=20
 

# # Trained on LJS VQ3 (01000-01100-01110)
# exp_dir="./exps/finished/ljs_code_mdl-01000-01100-01110_vq3_run1"
# exp_name=$(basename ${exp_dir})
# 
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/ljs \
#   --filelist="./filelists/rdvq_01000_01100_01110/ljs_audio_text_test_filelist.txt"
 

# Trained on LJS VQ3 (01000-01100) + chunk
exp_dir="./exps/finished/ljs_code_mdl-01000-01100_vq3_chunk_run1"
exp_name=$(basename ${exp_dir})

python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}/spokencoco \
  --filelist="./filelists/rdvq_01000_01100/spokencoco_tt.txt" \
  --num_utts=20

# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/ljs
# 
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/vctk \
#   --filelist="./filelists/rdvq_01000_01100/vctk_trimmed_valid_filelist.txt" \
#   --num_utts=20
#  
# python inference.py \
#   --ckpt_path=${exp_dir}/checkpoint_${step} \
#   --ckpt_args_path=${exp_dir}/args.json \
#   --out_dir=${eval_dir}/${exp_name}_step${step}/places \
#   --filelist="./filelists/rdvq_01000_01100/places_eng_400k_val_filelist.txt" \
#   --num_utts=20


# Trained on VCTK VQ3 (01000-01100)
# step=220000
# exp_dir="./exps/finished/vctk_code_mdl-01000-01100_vq3_chunk_spkemb_run1"
# exp_name=$(basename ${exp_dir})
# 
# uk_spks="12 47 86 44 43"
# us_spks="75 96 70 88 74"
# for label in $uk_spks $us_spks; do
#   python inference.py \
#     --ckpt_path=${exp_dir}/checkpoint_${step} \
#     --ckpt_args_path=${exp_dir}/args.json \
#     --out_dir=${eval_dir}/${exp_name}_step${step}/vctk_speaker_${label} \
#     --label=$label \
#     --filelist="./filelists/rdvq_01000_01100/vctk_trimmed_valid_filelist.txt" \
#     --num_utts=20
# 
#   python inference.py \
#     --ckpt_path=${exp_dir}/checkpoint_${step} \
#     --ckpt_args_path=${exp_dir}/args.json \
#     --out_dir=${eval_dir}/${exp_name}_step${step}/ljs_speaker_${label} \
#     --label=$label \
#     --filelist="./filelists/rdvq_01000_01100/ljs_audio_text_test_filelist.txt"
# done
 

# # Trained on Flickr8k VQ3 (01000-01100)
# step=131000
# exp_dir="./exps/finished/flickr8k_code_mdl-01000-01100_vq3_chunk_spkemb_fastloader_run2"
# exp_name=$(basename ${exp_dir})
# 
# spks="125 63 13 96 46 35"
# for label in $spks; do
#   python inference.py \
#     --ckpt_path=${exp_dir}/checkpoint_${step} \
#     --ckpt_args_path=${exp_dir}/args.json \
#     --out_dir=${eval_dir}/${exp_name}_step${step}/vctk_speaker_${label} \
#     --label=$label \
#     --filelist="./filelists/rdvq_01000_01100/flickr8k_tt.txt" \
#     --num_utts=20
# 
#   python inference.py \
#     --ckpt_path=${exp_dir}/checkpoint_${step} \
#     --ckpt_args_path=${exp_dir}/args.json \
#     --out_dir=${eval_dir}/${exp_name}_step${step}/spokencoco_speaker_${label} \
#     --filelist="./filelists/rdvq_01000_01100/spokencoco_tt.txt" \
#     --num_utts=20
# 
#   python inference.py \
#     --ckpt_path=${exp_dir}/checkpoint_${step} \
#     --ckpt_args_path=${exp_dir}/args.json \
#     --out_dir=${eval_dir}/${exp_name}_step${step}/ljs_speaker_${label} \
#     --label=$label \
#     --filelist="./filelists/rdvq_01000_01100/ljs_audio_text_test_filelist.txt"
# done
 

# # Compare different checkpoints
# for step in $(seq 1000 1000 20000); do
#   exp_dir="./exps/finished/code_mdl-01000-01100_vq3_ljs_run1"
#   exp_name=$(basename ${exp_dir})
#   python inference.py \
#     --ckpt_path=${exp_dir}/checkpoint_${step} \
#     --ckpt_args_path=${exp_dir}/args.json \
#     --out_dir=${eval_dir}/${exp_name}_step${step}
#   
#   exp_dir="./exps/finished/code_mdl-01000-01100_vq3_chunk_ljs_run1"
#   exp_name=$(basename ${exp_dir})
#   python inference.py \
#     --ckpt_path=${exp_dir}/checkpoint_${step} \
#     --ckpt_args_path=${exp_dir}/args.json \
#     --out_dir=${eval_dir}/${exp_name}_step${step}
# done

