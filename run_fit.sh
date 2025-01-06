#!/bin/bash
#SBATCH --job-name=CVSIM_Brain3_1
#SBATCH --output=CVSIM_Brain.out
#SBATCH --error=CVSIM_Brain.err
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --time=6:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=@student.uva.nl
#SBATCH -o %x-%j.out  # send stdout to outfile
#SBATCH -e %x-%j.err  # send stderr to errfile

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
# Activate the virtual environment
source ~/myenv/bin/activate

# Run the Python script
python Fit_Doppler.py