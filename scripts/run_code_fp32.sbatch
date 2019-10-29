#!/bin/bash 
#SBATCH --partition=sm
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --open-mode=append
#SBATCH --mem=8000
#SBATCH --job-name="t2-code"
#SBATCH --output=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs/t2-code-%j.out

. env.sh
export TORCH_HOME=/data/sls/temp/wnhsu/torch_home

logdir=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs

set -eux

expdir=$1
bs=$2
tr_file=$3
dt_file=$4
code_key=$5
code_dict=$6
n_symbols=$7
collapse_code=$8
other_hparams=$9
other_args=${10}

hparams="batch_size=${bs},fp16_run=False,distributed_run=True,text_or_code=code"
hparams="$hparams,training_files=$tr_file,validation_files=$dt_file"
hparams="$hparams,code_key=$code_key,code_dict=$code_dict"
hparams="$hparams,n_symbols=$n_symbols,collapse_code=$collapse_code"
if [ ! -z $other_hparams ]; then
  hparams="${hparams},${other_hparams}"
fi

echo "DIRECTORY: $expdir"
[ -d "$expdir" ] || mkdir -p "$expdir"
echo "$SLURM_JOB_ID" >> "$expdir/slurm_job_ids"
[ -f "$expdir/${SLURM_JOB_ID}.out"  ] || \
  ln -s "$logdir/t2-code-${SLURM_JOB_ID}.out" "$expdir/${SLURM_JOB_ID}.out"

python -m multiproc train.py -o $expdir -l log --hparams="$hparams" $other_args
