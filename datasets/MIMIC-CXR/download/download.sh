#!/bin/bash
#SBATCH --job-name=download_mimic_cxr
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

for f in p10_part_*; do
  nohup wget -r -N -c -np -nH --cut-dirs=1 \
    --user xxx --password xxx \
    -i "$f" \
    --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/ &
done

