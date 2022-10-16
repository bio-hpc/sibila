folder=$1
endjob=$2
partition=$3

# send individual jobs for each interpretability method, model and data block
jobs_ids=""
for job in `find ${PWD}/${folder}/jobs/*.sh -type f ! -empty`
do
    jid=$(sbatch ${job} | cut -d ' ' -f 4)
    if [ -z "${jobs_ids}" ]; then
        jobs_ids="${jid}"
    else
        jobs_ids="${jobs_ids}:${jid}"
    fi
done

# send final job for building documents and compressing files
export_var="PARTITION=${PARTITION},TIME=${TIME},MEM=${MEM},SINGULARITY=${SINGULARITY},PYTHON_RUN=${PYTHON_RUN},IMG_SINGULARITY=${IMG_SINGULARITY}"
sbatch --export=${export_var} --dependency=afterany:${jobs_ids} ${endjob}
