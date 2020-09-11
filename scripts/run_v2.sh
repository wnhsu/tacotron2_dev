#!/bin/bash 

set -xu

# # VQ3 (01000-01100-01110)
# filelist_root="./filelists/rdvq_01000_01100_01110"
# tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
# dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/ljspeech/wav_h5py"
# tr_h5="$h5_root/ljs_audio_text_train_filelist_wav.h5"
# dt_h5="$h5_root/ljs_audio_text_val_filelist_wav.h5"
# tr_idx="$h5_root/ljs_audio_text_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/ljs_audio_text_val_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
# args="dist_url=tcp://localhost:54324,num_workers=8"
# sbatch ./scripts/run_code_fp16_fastloader.sbatch \
#   "./exps/ljs_code_mdl-01000-01100-01110_vq3_bs72_run1" 72 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""


###### Compare batch size (fixed steps)

# VQ3 (01000-01100)
# filelist_root="./filelists/rdvq_01000_01100"
# tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
# dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/ljspeech/wav_h5py"
# tr_h5="$h5_root/ljs_audio_text_train_filelist_wav.h5"
# dt_h5="$h5_root/ljs_audio_text_val_filelist_wav.h5"
# tr_idx="$h5_root/ljs_audio_text_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/ljs_audio_text_val_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
# args="dist_url=tcp://localhost:54324,num_workers=4,epochs=500"
# sbatch ./scripts/run_code_fp16_fastloader_1gpu.sbatch \
#   "./exps/ljs_code_mdl-01000-01100_vq3_bs64" 64 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""
# 
# 
# 
# filelist_root="./filelists/rdvq_01000_01100"
# tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
# dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/ljspeech/wav_h5py"
# tr_h5="$h5_root/ljs_audio_text_train_filelist_wav.h5"
# dt_h5="$h5_root/ljs_audio_text_val_filelist_wav.h5"
# tr_idx="$h5_root/ljs_audio_text_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/ljs_audio_text_val_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
# args="dist_url=tcp://localhost:54325,num_workers=4,epochs=1000"
# sbatch ./scripts/run_code_fp16_fastloader_2gpu.sbatch \
#   "./exps/ljs_code_mdl-01000-01100_vq3_bs128" 64 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""
# 
# 
# 
# filelist_root="./filelists/rdvq_01000_01100"
# tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
# dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/ljspeech/wav_h5py"
# tr_h5="$h5_root/ljs_audio_text_train_filelist_wav.h5"
# dt_h5="$h5_root/ljs_audio_text_val_filelist_wav.h5"
# tr_idx="$h5_root/ljs_audio_text_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/ljs_audio_text_val_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
# args="dist_url=tcp://localhost:54326,num_workers=4,epochs=2000"
# sbatch ./scripts/run_code_fp16_fastloader_4gpu.sbatch \
#   "./exps/ljs_code_mdl-01000-01100_vq3_bs256" 64 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""


# filelist_root="./filelists/rdvq_01000_01100"
# tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
# dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/ljspeech/wav_h5py"
# tr_h5="$h5_root/ljs_audio_text_train_filelist_wav.h5"
# dt_h5="$h5_root/ljs_audio_text_val_filelist_wav.h5"
# tr_idx="$h5_root/ljs_audio_text_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/ljs_audio_text_val_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+3  }')
# args="dist_url=tcp://localhost:54327,num_workers=4,epochs=500,add_sos=True,add_eos=True"
# sbatch ./scripts/run_code_fp16_fastloader_2gpu.sbatch \
#   "./exps/ljs_code_mdl-01000-01100_vq3_bs64_sos_eos" 32 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""


filelist_root="./filelists/rdvq_01000_01100"
tr_file="$filelist_root/ljs_audio_text_train_filelist.txt"
dt_file="$filelist_root/ljs_audio_text_val_filelist.txt"
h5_root="/data/sls/temp/wnhsu/data/tts/ljspeech/wav_h5py"
tr_h5="$h5_root/ljs_audio_text_train_filelist_wav.h5"
dt_h5="$h5_root/ljs_audio_text_val_filelist_wav.h5"
tr_idx="$h5_root/ljs_audio_text_train_filelist_wav2idx.txt"
dt_idx="$h5_root/ljs_audio_text_val_filelist_wav2idx.txt"
code_key=code_quant2
code_dict=$filelist_root/code_dict_quant2
n_symbols=$(wc -l $code_dict | awk '{ print $1+3  }')
args="dist_url=tcp://localhost:54327,num_workers=4,epochs=500,add_sos=True,add_eos=True"
sbatch ./scripts/run_code_fp16_fastloader_2gpu.sbatch \
  "./exps/ljs_code_mdl-01000-01100_vq2_bs64_sos_eos" 32 \
  $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
  $code_key $code_dict $n_symbols True \
  $args ""



##### VCTK, filter at 8sec, removed 181 utts (0.42%)

# # Speaker Embedding + VQ3 (01000-01100) + fastloader
# filelist_root="./filelists/rdvq_01000_01100"
# tr_file="$filelist_root/vctk_trimmed_train_filelist.txt"
# dt_file="$filelist_root/vctk_trimmed_valid_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/vctk_trimmed/wav_16bit_signed_h5py"
# tr_h5="$h5_root/vctk_trimmed_train_filelist_wav.h5"
# dt_h5="$h5_root/vctk_trimmed_valid_filelist_wav.h5"
# tr_idx="$h5_root/vctk_trimmed_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/vctk_trimmed_valid_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
# args="max_wav_len=8,obs_dim=256,obs_label_key=speaker,obs_n_class=108"
# args="$args,obs_label_dict=./filelists/original/vctk_speakers.txt"
# args="$args,epochs=250,dist_url=tcp://localhost:54322,num_workers=8"
# sbatch ./scripts/run_code_fp16_fastloader_2gpu.sbatch \
#   "./exps/vctk_code_mdl-01000-01100_vq3_spkemb256_bs64" 32 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""

# filelist_root="./filelists/rdvq_01000_01100"
# tr_file="$filelist_root/vctk_trimmed_train_filelist.txt"
# dt_file="$filelist_root/vctk_trimmed_valid_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/vctk_trimmed/wav_16bit_signed_h5py"
# tr_h5="$h5_root/vctk_trimmed_train_filelist_wav.h5"
# dt_h5="$h5_root/vctk_trimmed_valid_filelist_wav.h5"
# tr_idx="$h5_root/vctk_trimmed_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/vctk_trimmed_valid_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+3  }')
# args="max_wav_len=8,obs_dim=256,obs_label_key=speaker,obs_n_class=108"
# args="$args,obs_label_dict=./filelists/original/vctk_speakers.txt"
# args="$args,epochs=250,dist_url=tcp://localhost:54323,num_workers=8"
# args="$args,add_sos=True,add_eos=True"
# sbatch ./scripts/run_code_fp16_fastloader_2gpu.sbatch \
#   "./exps/vctk_code_mdl-01000-01100_vq3_spkemb256_bs64_sos_eos" 32 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""


# filelist_root="./filelists/rdvq_01000_01100"
# tr_file="$filelist_root/vctk_trimmed_train_filelist.txt"
# dt_file="$filelist_root/vctk_trimmed_valid_filelist.txt"
# h5_root="/data/sls/temp/wnhsu/data/tts/vctk_trimmed/wav_16bit_signed_h5py"
# tr_h5="$h5_root/vctk_trimmed_train_filelist_wav.h5"
# dt_h5="$h5_root/vctk_trimmed_valid_filelist_wav.h5"
# tr_idx="$h5_root/vctk_trimmed_train_filelist_wav2idx.txt"
# dt_idx="$h5_root/vctk_trimmed_valid_filelist_wav2idx.txt"
# code_key=code_quant3
# code_dict=$filelist_root/code_dict_quant3
# n_symbols=$(wc -l $code_dict | awk '{ print $1+1  }')
# args="max_wav_len=8,obs_dim=128,obs_label_key=speaker,obs_n_class=108"
# args="$args,obs_label_dict=./filelists/original/vctk_speakers.txt"
# args="$args,epochs=250,dist_url=tcp://localhost:54321,num_workers=8"
# sbatch ./scripts/run_code_fp16_fastloader_2gpu.sbatch \
#   "./exps/vctk_code_mdl-01000-01100_vq3_spkemb128_bs64" 32 \
#   $tr_file $tr_h5 $tr_idx $dt_file $dt_h5 $dt_idx \
#   $code_key $code_dict $n_symbols True \
#   $args ""


