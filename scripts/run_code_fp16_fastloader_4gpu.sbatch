#!/bin/bash 
#SBATCH --partition=titanx
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --open-mode=append
#SBATCH --mem=120000
#SBATCH --job-name="t2-code"
#SBATCH --output=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs/t2-code-%j.out

. env.sh
export TORCH_HOME=/data/sls/temp/wnhsu/torch_home

logdir=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs

set -eux

expdir=$1
bs=$2
tr_file=$3
tr_h5=$4
tr_idx=$5
dt_file=$6
dt_h5=$7
dt_idx=$8
code_key=$9
code_dict=${10}
n_symbols=${11}
collapse_code=${12}
other_hparams=${13}
other_args=${14}

# make local copy for wav_h5
echo "time before copy wav_h5 : $(date)"
local_dir="/scratch/wnhsu/data"
local_tr_h5=$local_dir/$(basename $tr_h5)
local_dt_h5=$local_dir/$(basename $dt_h5)
mkdir -p $local_dir
[ -f "$local_tr_h5" ] || (echo "copying $tr_h5" && cp $tr_h5 $local_tr_h5)
[ -f "$local_dt_h5" ] || (echo "copying $dt_h5" && cp $dt_h5 $local_dt_h5)
echo "time after copy wav_h5 : $(date)"

# set hyperparameters
hparams="batch_size=${bs},fp16_run=True,distributed_run=True,text_or_code=code"
hparams="$hparams,training_files=$tr_file"
hparams="$hparams,training_wav_h5=$local_tr_h5"
hparams="$hparams,training_wav_path2idx=$tr_idx"
hparams="$hparams,validation_files=$dt_file"
hparams="$hparams,validation_wav_h5=$local_dt_h5"
hparams="$hparams,validation_wav_path2idx=$dt_idx"
hparams="$hparams,code_key=$code_key,code_dict=$code_dict"
hparams="$hparams,n_symbols=$n_symbols,collapse_code=$collapse_code"
if [ ! -z $other_hparams ]; then
  hparams="${hparams},${other_hparams}"
fi

echo "DIRECTORY: $expdir"
[ -d "$expdir" ] || mkdir -p "$expdir"
echo "$SLURM_JOB_ID" >> "$expdir/slurm_job_ids"
[ -L "$expdir/${SLURM_JOB_ID}.out"  ] || \
  ln -s "$logdir/t2-code-${SLURM_JOB_ID}.out" "$expdir/${SLURM_JOB_ID}.out"

python -m multiproc train.py -o $expdir -l log --hparams="$hparams" $other_args
