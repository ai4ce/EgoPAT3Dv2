#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=44:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=TE_d1_unseen
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yc6317@nyu.edu
#SBATCH --output=%x.out

module purge
module load openmpi/intel/4.0.5

singularity exec --nv \
	    --overlay ../overlay_1.ext3:ro \
	    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh;         
		mpirun python test_DDP.py --model_epoch cn_t_post_hand_newloss_1/d1_unseen_epoch_30 --checkpoint ./experiment/cn_t_post_hand_newloss_1/checkpoints/cn_t_post_hand_newloss_1-0.000000-0030.pth --config_file ./configs/cn_t_post_hand_newloss_1_d1_seen.yaml > ./experiment/cn_t_post_hand_newloss_1/eval/output_logs/d1_unseen_30.log 2>&1 &
        job1=$!
		mpirun python test_DDP.py --model_epoch cn_t_post_hand_newloss_1/d1_unseen_epoch_10 --checkpoint ./experiment/cn_t_post_hand_newloss_1/checkpoints/cn_t_post_hand_newloss_1-0.000000-0010.pth --config_file ./configs/cn_t_post_hand_newloss_1_d1_seen.yaml > ./experiment/cn_t_post_hand_newloss_1/eval/output_logs/d1_unseen_10.log 2>&1 &
		job2=$!
		mpirun python test_DDP.py --model_epoch cn_t_post_hand_newloss_1/d1_unseen_epoch_11 --checkpoint ./experiment/cn_t_post_hand_newloss_1/checkpoints/cn_t_post_hand_newloss_1-0.000000-0011.pth --config_file ./configs/cn_t_post_hand_newloss_1_d1_seen.yaml > ./experiment/cn_t_post_hand_newloss_1/eval/output_logs/d1_unseen_11.log 2>&1 &
		job3=$!

		wait $job1 $job2 $job3 

        python eval.py --model_name 'cn_t_post_hand_newloss_1'
		"