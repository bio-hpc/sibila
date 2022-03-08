#!/bin/sh
#
# Parameters job
#
CPUS=1
TIME=24:00:00
NAME_JOB="ML_GS"
#MEM=2000M
PROJECT="" #at the moment it is not used
#
#  Constans
#
QUEUE_MANAGER="SLURM"
PYTHON_RUN="python3"
SIBILA="sibila.py"
MODEL_BUILDER="Scripts/GridSearch/model_builder.py"
RESULT_ANALYZER="Scripts/ResultAnalyzer.py"
SCRIPT_QUEUE="Tools/Bash/Queue_manager/${QUEUE_MANAGER}.sh"
CMD_QUEUE="sbatch"
IMG_SINGULARITY="Tools/Singularity/sibila.simg"
CMD_SING="singularity exec ${IMG_SINGULARITY} ${PYTHON_RUN} ${SIBILA}"
CMD_SING_SEARCH="singularity exec ${IMG_SINGULARITY} ${PYTHON_RUN} ${MODEL_BUILDER}"
CMD_SING_EXCEL="singularity exec ${IMG_SINGULARITY} ${PYTHON_RUN} ${RESULT_ANALYZER}"
CMD_SING_HELP="singularity exec ${IMG_SINGULARITY} ${PYTHON_RUN} ${SIBILA}"
SINGULARITY=true
PARAM_MULTIJOB_JOB="-nj" #optional parameter to add to the folder name and job

multi_job=""
params=""
regression=""

while [ ${#} -gt 0 ];do

  if [ "${1}" = "-g" ]; then
    shift
    folder_gs=${1}
    if [ -d "${1}" ]; then
      rm -rf ${1}
    fi
    mkdir ${1}
    shift
  elif [ "${1}" = "-o" ]; then
    params=${params}" "${1}
    shift
    model=${1}
  elif [ "${1}" = "-r" ]; then
    params=${params}" "${1}
    regression="${1}"
    shift
  elif [ "${1}" = "-h" ];then
    $cmd_help $1
    echo "-nj optional parameter to add to the folder name and job"
    exit
  else
    params=${params}" "$1
    shift
  fi
done

# generate models for grid search
job_gs=${PWD}/${folder_gs}/job_grid.sh
cmd_run="${CMD_SING_SEARCH}"
sh $SCRIPT_QUEUE "${PWD}/${folder_gs}/" ${NAME_JOB} ${TIME} ${CPUS} ${MEM} > ${job_gs}
echo "${cmd_run} -d ${folder_gs} -m ${model}"                              >> ${job_gs}
gs_id=$(${CMD_QUEUE} ${job_gs} | awk '{print $NF}')

# summarize results in an excel file
job_xlsx=${PWD}/${folder_gs}/job_xlsx.sh
cmd_run="${CMD_SING_EXCEL}"
NAME_JOB="ML_XLSX"
sh $SCRIPT_QUEUE "${PWD}/${folder_gs}/" ${NAME_JOB} ${TIME} ${CPUS} ${MEM}   > ${job_xlsx}
echo "${cmd_run} -d . -o ${model}-$(date +%Y%m%d-%H%M%S).xlsx ${regression}" >> ${job_xlsx}

# call a second script to launch all the models
job_all=${PWD}/${folder_gs}/job_all.sh
NAME_JOB="ML_LAI_ALL"
sh $SCRIPT_QUEUE "${PWD}/${folder_gs}/" ${NAME_JOB} ${TIME} ${CPUS} ${MEM} > ${job_all}
echo "dependency=\"--dependency=after\""                                   >> ${job_all}
echo "for foo in ${folder_gs}/*.json"                                      >> ${job_all}
echo "do"                                                                  >> ${job_all}
echo "  model_id=\$(basename \${foo})"                                     >> ${job_all}
echo "  model_id=\$(echo \${model_id%.*})"                                 >> ${job_all}
echo "  job_lai=${PWD}/${folder_gs}/job_lai_\${model_id}.sh"               >> ${job_all}
echo "  job_id=\$(sh sibila.sh ${params} -p \${foo} -f \${model_id} | awk '{print \$NF}')" >> ${job_all}
echo "  dependency=\"\${dependency}\":\${job_id}"                          >> ${job_all}
echo "done"                                                                >> ${job_all}
echo "${CMD_QUEUE} \${dependency} ${job_xlsx}"                             >> ${job_all}

${CMD_QUEUE} --dependency=afterok:${gs_id} ${job_all}
