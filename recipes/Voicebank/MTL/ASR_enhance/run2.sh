#!/bin/bash

# We jump into the submission dir
cd ${SLURM_SUBMIT_DIR}

# Activate the virtual env
source /netscratch/sagar/thesis/sb_env/bin/activate

# install necessary packages
apt-get update
apt-get -y install libsndfile1

# Reset BASH time counter
SECONDS=0

# training command: e.g. train.py hparams/train.yaml
# training_command=" \
# train.py hparams/enhance_mimic.yaml \
# --seed 123 \
# --input_type clean_noisy_mix \
# --results_folder baseline_2_single_gpu \
# --pretrain_perceptual_path baseline_2_single_gpu/perceptual_model/123/save/CKPT+2022-07-01+22-19-05+00/src_embedding.ckpt
# "

training_command="
train.py  hparams/robust_asr.yaml \
--seed 223 \
--target_type phones \
--input_type clean_noisy_mix \
--results_folder baseline_2 \
--pretrained_enhance_path baseline_2/enhance_model/123/save/CKPT+2022-07-04+10-26-02+00/
"

NUM_GPUS=`python -c "import torch; print(torch.cuda.device_count())"`
# cmd="python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${SLURM_NODEID} ${training_command} --distributed_launch"
cmd="python ${training_command}"
echo "Running: ${cmd}"
${cmd}

# Display total runtime
ELAPSED="Total runtime: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ${ELAPSED}

