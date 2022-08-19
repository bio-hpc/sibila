echo "#!/bin/bash"
echo "#SBATCH --output=${1}job.out"
echo "#SBATCH --error=${1}job.err"
echo "#SBATCH -J ${2}"
echo "#SBATCH --time=${3}"
echo "#SBATCH --cpus-per-task=${4}"
if [ -n "${5}" ]; then
    echo "#SBATCH --mem=${5}"
fi
#SBATCH -p standard

