#!/bin/bash 

step=77000
eval_dir="./exps/eval"

exp_dir="./exps/finished/basic_ljs_run1"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}

exp_dir="./exps/finished/code_mdl-01000-01100_vq3_ljs_run1"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}

exp_dir="./exps/finished/code_mdl-01000-01100_vq3_ljs_run1"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}_vctk \
  --filelist="./filelists/rdvq_01000_01100/vctk_trimmed_valid_filelist.txt" \
  --num_utts=20
 
exp_dir="./exps/finished/code_mdl-01000-01100_vq3_ljs_run1"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}_places \
  --filelist="./filelists/rdvq_01000_01100/places_eng_400k_val_filelist.txt" \
  --num_utts=20
 
exp_dir="./exps/finished/code_mdl-01000-01100_vq3_chunk_ljs_run1"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}

exp_dir="./exps/finished/code_mdl-01000-01100_vq3_chunk_ljs_run1"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}_vctk \
  --filelist="./filelists/rdvq_01000_01100/vctk_trimmed_valid_filelist.txt" \
  --num_utts=20
 
exp_dir="./exps/finished/code_mdl-01000-01100_vq3_chunk_ljs_run1"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}_places \
  --filelist="./filelists/rdvq_01000_01100/places_eng_400k_val_filelist.txt" \
  --num_utts=20

exp_dir="./exps/finished/code_mdl-01000-01100_vq2_ljs_run1/"
exp_name=$(basename ${exp_dir})
python inference.py \
  --ckpt_path=${exp_dir}/checkpoint_${step} \
  --ckpt_args_path=${exp_dir}/args.json \
  --out_dir=${eval_dir}/${exp_name}_step${step}

for step in $(seq 1000 1000 20000); do
  exp_dir="./exps/finished/code_mdl-01000-01100_vq3_ljs_run1"
  exp_name=$(basename ${exp_dir})
  python inference.py \
    --ckpt_path=${exp_dir}/checkpoint_${step} \
    --ckpt_args_path=${exp_dir}/args.json \
    --out_dir=${eval_dir}/${exp_name}_step${step}
  
  exp_dir="./exps/finished/code_mdl-01000-01100_vq3_chunk_ljs_run1"
  exp_name=$(basename ${exp_dir})
  python inference.py \
    --ckpt_path=${exp_dir}/checkpoint_${step} \
    --ckpt_args_path=${exp_dir}/args.json \
    --out_dir=${eval_dir}/${exp_name}_step${step}
done

