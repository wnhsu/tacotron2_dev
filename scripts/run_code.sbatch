#!/bin/bash 
#SBATCH --partition=titanx,2080
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH -n 1
#SBATCH -x sls-2080-2
#SBATCH --open-mode=append
#SBATCH --mem=8000
#SBATCH --job-name="t2-code-ljs"
#SBATCH --output=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs/t2-code-ljs-%j.out

. env.sh
export TORCH_HOME=/data/sls/temp/wnhsu/torch_home

logdir=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs

set -eux

expdir=$1
bs=$2
filedir=$3
code_key=$4
code_dict=$5
n_symbols=$6
collapse_code=$7
other_hparams=$8
other_args=$9

args="batch_size=${bs},fp16_run=True,distributed_run=True,text_or_code=code"
args="$args,training_files=$filedir/ljs_audio_text_train_filelist.txt"
args="$args,validation_files=$filedir/ljs_audio_text_val_filelist.txt"
args="$args,code_key=$code_key,code_dict=$code_dict"
args="$args,n_symbols=$n_symbols,collapse_code=$collapse_code"
if [ ! -z $other_hparams ]; then
  args="${args},${other_hparams}"
fi

echo "DIRECTORY: $expdir"
[ -d "$expdir" ] || mkdir -p "$expdir"
echo "$SLURM_JOB_ID" >> "$expdir/slurm_job_ids"
[ -f "$expdir/${SLURM_JOB_ID}.out"  ] || \
  ln -s "$logdir/t2-code-ljs-${SLURM_JOB_ID}.out" "$expdir/${SLURM_JOB_ID}.out"

python -m multiproc train.py -o $expdir -l log --hparams="$args" $other_args
