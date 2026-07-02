#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — single RTX 4090, models up to ~14B)
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name tsr_gpu
#SBATCH --output logs_terminal/tsr_gpu_%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=60G

### Print debug info ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "current path:\t" $PWD "\n\n"

### Start code ###
module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate multits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Using python from: $(which python)"
python -c "import torch; print('Success! Torch version:', torch.__version__)"

python run_exp.py "$@"
