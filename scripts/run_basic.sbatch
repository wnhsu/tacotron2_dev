#!/bin/bash 
#SBATCH --partition=titanx,2080
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --open-mode=append
#SBATCH --mem=8000
#SBATCH --job-name="t2-basic-ljs"
#SBATCH --output=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs/t2-basic-ljs-%j.out

. env.sh
export TORCH_HOME=/data/sls/temp/wnhsu/torch_home

logdir=/data/sls/u/wnhsu/code/tacotron2_factory/tacotron2_20191017/logs

set -eux

expdir=$1
bs=$2
other_hparams=$3
other_args=$4

hparams="batch_size=${bs},fp16_run=True,distributed_run=True"
if [ ! -z $other_hparams ]; then
  hparams="${hparams},${other_hparams}"
fi

echo "DIRECTORY: $expdir"
[ -d "$expdir" ] || mkdir -p "$expdir"
echo "$SLURM_JOB_ID" >> "$expdir/slurm_job_ids"
[ -f "$expdir/${SLURM_JOB_ID}.out"  ] || \
  ln -s "$logdir/t2-basic-ljs-${SLURM_JOB_ID}.out" "$expdir/${SLURM_JOB_ID}.out"

python -m multiproc train.py -o $expdir -l log --hparams="$hparams" $other_args
