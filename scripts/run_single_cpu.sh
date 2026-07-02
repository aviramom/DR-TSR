#!/bin/bash

################################################################################################
### sbatch configuration parameters (CPU-only — baselines, API models)
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-00:30:00
#SBATCH --job-name tsr_cpu
#SBATCH --output logs_terminal/tsr_cpu_%J.out
#SBATCH --mem=16G

### Print debug info ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "current path:\t" $PWD "\n\n"

### Start code ###
module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate multits

echo "Using python from: $(which python)"
python run_exp.py "$@"
