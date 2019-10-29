#!/bin/bash 

set -xu

####################
# LJSpeech
####################

# Char
sbatch ./scripts/run_basic.sbatch "./exps/basic_ljs_run1" 40 ""

# VQ3 (01000-01100)
filelist_root="./filelists/rdvq_01000_01100"
tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
code_key=code_quant3
code_dict=$filelist_root/code_dict_quant3
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
sbatch ./scripts/run_code.sbatch \
  "./exps/code_mdl-01000-01100_vq3_ljs_run1" 40 \
  $tr_file $dt_file $code_key $code_dict $n_symbols True \
  "dist_url=tcp://localhost:54322" ""

# VQ2 (01000-01100)
filelist_root="./filelists/rdvq_01000_01100"
tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
code_key=code_quant2
code_dict=$filelist_root/code_dict_quant2
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
sbatch ./scripts/run_code.sbatch \
  "./exps/code_mdl-01000-01100_vq2_ljs_run1" 40 \
  $tr_file $dt_file $code_key $code_dict $n_symbols True \
  "dist_url=tcp://localhost:54328" ""

# VQ2 (conti-01000)
filelist_root="./filelists/rdvq_conti_01000"
tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
code_key=code_quant2
code_dict=$filelist_root/code_dict_quant2
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
sbatch ./scripts/run_code_fp32.sbatch \
  "./exps/code_mdl-conti-01000_vq2_ljs_run1" 24 \
  $tr_file $dt_file $code_key $code_dict $n_symbols True \
  "dist_url=tcp://localhost:54329" ""

# Chunk + VQ3 (01000-01100)
filelist_root="./filelists/rdvq_01000_01100"
tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
code_key=code_quant3
code_dict=$filelist_root/code_dict_quant3
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
args="dist_url=tcp://localhost:54323,chunk_code=True,init_chunk=50,chunk_incr=5"
sbatch ./scripts/run_code.sbatch \
  "./exps/code_mdl-01000-01100_vq3_chunk_ljs_run1" 40 \
  $tr_file $dt_file $code_key $code_dict $n_symbols True \
  $args ""

# Chunk + Always-Chunk (min=50) + VQ3 (01000-01100)
filelist_root="./filelists/rdvq_01000_01100"
tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
code_key=code_quant3
code_dict=$filelist_root/code_dict_quant3
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
args="dist_url=tcp://localhost:54323,chunk_code=True,init_chunk=50,chunk_incr=5"
args="$args,always_chunk=True,min_chunk=50"
sbatch ./scripts/run_code.sbatch \
  "./exps/code_mdl-01000-01100_vq3_chunk_always_ljs_run1" 40 \
  $tr_file $dt_file $code_key $code_dict $n_symbols True \
  $args ""

####################
# VCTK
####################

# Speaker Embedding
args="training_files=./filelists/vctk_trimmed_train_filelist.txt"
args="$args,validation_files=./filelists/vctk_trimmed_valid_filelist.txt"
args="$args,obs_label_key=speaker,obs_n_class=108"
args="$args,obs_label_dict=./filelists/vctk_speakers.txt"
sbatch ./scripts/run_basic_fp32.sbatch "./exps/spkemb_vctk_run1" 24 $args ""

# Lat=64, KL=1.0
args="training_files=./filelists/vctk_trimmed_train_filelist.txt"
args="$args,validation_files=./filelists/vctk_trimmed_valid_filelist.txt"
args="$args,lat_dim=64,dist_url=tcp://localhost:54324"
sbatch ./scripts/run_basic_fp32.sbatch "./exps/lat64_vctk_run1" 24 $args ""

# Lat=64, KL=0
args="training_files=./filelists/vctk_trimmed_train_filelist.txt"
args="$args,validation_files=./filelists/vctk_trimmed_valid_filelist.txt"
args="$args,lat_dim=64,dist_url=tcp://localhost:54325,kld_weight=0"
sbatch ./scripts/run_basic_fp32.sbatch "./exps/lat64_klw0_vctk_run1" 24 $args ""

# Lat=64, KL=0.01
args="training_files=./filelists/vctk_trimmed_train_filelist.txt"
args="$args,validation_files=./filelists/vctk_trimmed_valid_filelist.txt"
args="$args,lat_dim=64,dist_url=tcp://localhost:54326,kld_weight=0.01"
sbatch ./scripts/run_basic_fp32.sbatch "./exps/lat64_klw0.01_vctk_run1" 24 $args ""

####################
# Places English 400k
####################

tr_file="./filelists/rdvq_01000_01100/places_eng_400k_train_filelist.txt"
dt_file="./filelists/rdvq_01000_01100/places_eng_400k_val100_filelist.txt"
code_key=code_quant3
code_dict="./filelists/rdvq_01000_01100/code_dict_quant3"
n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
args="chunk_code=True,init_chunk=50,chunk_incr=50,max_chunk=250"
args="$args,lat_dim=32,max_wav_len=20.48,dist_url=tcp://localhost:54323"
sbatch ./scripts/run_code_v2_fp32.sbatch \
  "./exps/code_mdl-01000-01100_vq3_chunk_places400eng_run1" 24 \
  $tr_file $dt_file $code_key $code_dict $n_symbols True $args ""

