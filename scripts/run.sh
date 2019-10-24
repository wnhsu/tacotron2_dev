#!/bin/bash 

set -x

sbatch ./scripts/run_basic.sbatch "./exps/basic_ljs_run1" 40 ""

filelist_root="./filelists/rdvq_01000_01100"
code_key=code_quant3
code_dict=$filelist_root/code_dict_quant3
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
sbatch ./scripts/run_code.sbatch \
  "./exps/code_mdl-01000-01100_vq3_ljs_run1" 40 \
  $filelist_root $code_key $code_dict $n_symbols True \
  "dist_url=tcp://localhost:54322" \
  "-c ./exps/code_mdl-01000-01100_vq3_ljs_run1/checkpoint_45000"

filelist_root="./filelists/rdvq_01000_01100"
code_key=code_quant2
code_dict=$filelist_root/code_dict_quant2
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
sbatch ./scripts/run_code.sbatch \
  "./exps/code_mdl-01000-01100_vq2_ljs_run1" 40 \
  $filelist_root $code_key $code_dict $n_symbols True \
  "dist_url=tcp://localhost:54328" \
  "-c ./exps/code_mdl-01000-01100_vq2_ljs_run1/checkpoint_28000"

filelist_root="./filelists/rdvq_conti_01000"
code_key=code_quant2
code_dict=$filelist_root/code_dict_quant2
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
sbatch ./scripts/run_code_fp32.sbatch \
  "./exps/code_mdl-conti-01000_vq2_ljs_run1" 24 \
  $filelist_root $code_key $code_dict $n_symbols True \
  "dist_url=tcp://localhost:54329"
