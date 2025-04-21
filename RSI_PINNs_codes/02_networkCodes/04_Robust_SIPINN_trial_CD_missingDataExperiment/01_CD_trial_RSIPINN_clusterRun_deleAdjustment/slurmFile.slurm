#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=iist
#SBATCH -o runLog-%j

# runnig python script
echo "startTime" > log.executionTime
date >> log.executionTime
python script_RSIPINN.py > log.pythonRun 2>&1
echo "endTime" >> log.executionTime
date >> log.executionTime
