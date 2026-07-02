#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — 2× RTX 6000, 27B models)
### Uses multits_large env (newer transformers for Qwen3.6-27B / qwen3_5 architecture)
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name tsr_gpu_large
#SBATCH --output logs_terminal/tsr_gpu_large_%J.out
#SBATCH --gpus=rtx_6000:2
#SBATCH --exclude=ee-l40s-01
#SBATCH --mem=128G

### Print debug info ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "current path:\t" $PWD "\n\n"

### Start code ###
module load anaconda
module load cuda/12.4
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate multits_large
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "Using python from: $(which python)"
python -c "import torch; print('Success! Torch version:', torch.__version__)"

python run_exp.py "$@"
