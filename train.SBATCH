#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=44:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=TR_cn_t_post_hand_newloss_1
#SBATCH --partition=rtx8000,a100_2,a100_1,tandon_a100_2,tandon_a100_1,stake_a100_1,stake_a100_2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[replace with yours]
#SBATCH --output=%x.out

module purge

# export NCCL_DEBUG=INFO

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo $nodes
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
echo $SLURM_JOB_ID
echo $head_node_ip:29500

srun singularity exec --nv \
	    --overlay ../overlay_1.ext3:ro \
	    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; \
		torchrun --nnodes 4 \
		--nproc_per_node 1 \
		--rdzv_id $SLURM_JOB_ID \
		--rdzv_backend c10d \
		--rdzv_endpoint $head_node_ip:29500 \
		train_DDP.py --config_file ./configs/cn_t_post_hand_newloss_1_d1_seen.yaml"