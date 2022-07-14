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
training_command="
train.py hparams/robust_asr.yaml \
--target_type phones \
--seed 2428 \
--pretrained_enhance_path results/enhance_model/1288/save/CKPT+2022-07-04+10-19-15+00/
"

NUM_GPUS=`python -c "import torch; print(torch.cuda.device_count())"`
# cmd="python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${SLURM_NODEID} ${training_command} --distributed_launch"
cmd="python ${training_command}"
echo "Running: ${cmd}"
${cmd}

# Display total runtime
ELAPSED="Total runtime: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ${ELAPSED}
