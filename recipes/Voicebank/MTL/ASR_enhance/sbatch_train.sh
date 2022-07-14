#!/bin/bash

#SBATCH --nodes=1               # Number of nodes or servers. See: http://koeln.kl.dfki.de:3000/d/slurm-resources/resources?orgId=1&refresh=15s
#SBATCH --ntasks-per-node=1     # Number of task in each node 
#SBATCH --cpus-per-task=4       # We want 4 cores for this job.
#SBATCH --mem-per-cpu=16gb      # each core to have 16 Gb RAM
#SBATCH --gres=gpu:1            # We want 4 GPUs in each node for this job.
#SBATCH --time=30:00:00         # Run this task no longer that 30 hrs.
#SBATCH --partition=RTXA6000,A100,V100-32GB,RTX3090,V100-32GB # V100-16GB
#SBATCH --job-name=mimic_loss
#SBATCH --output=mimic_loss_%A.logs

echo "#############################"
date
echo "Current dir: " ${SLURM_SUBMIT_DIR}
echo "Hostname: `hostname`"

# Print the task details.
echo "Job ID: ${SLURM_JOBID}"
echo "SLURM array task ID:  ${SLURM_ARRAY_TASK_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}" 
echo "Cluster name: ${SLURM_CLUSTER_NAME}"
echo "Partition name: ${SLURM_JOB_PARTITION}" 
echo "Using: `which python`"
echo -e "#############################\n"

srun -K \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`",/home/$USER/:/home/$USER/ \
--container-image=/netscratch/sagar/docker_images/images/10.2.sqsh \
--container-workdir="`pwd`" \
bash run2.sh
