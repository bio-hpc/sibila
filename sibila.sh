#!/bin/bash
#
# Parameters job
#
CPUS=1
TIME=72:00:00
NAME_JOB="SIBILA"
MEM=1500M
PARTITION="slims"
PROJECT="" #at the moment it is not used
#
#  Constans
#
QUEUE_MANAGER="SLURM"
PYTHON_RUN="python3"
SIBILA="sibila.py"
SCRIPT_QUEUE="Tools/Bash/Queue_manager/${QUEUE_MANAGER}.sh"
CMD_QUEUE="sbatch"
CMD_EXEC="singularity exec --bind $PWD"
IMG_SINGULARITY="Tools/Singularity/sibila.sif"
TEST_CMD="${CMD_EXEC} ${IMG_SINGULARITY} python3 -m unittest discover"
CMD_SING="${CMD_EXEC} --nv ${IMG_SINGULARITY} ${PYTHON_RUN} ${SIBILA}"
CMD_SING_HELP="${CMD_EXEC} ${IMG_SINGULARITY} ${PYTHON_RUN} ${SIBILA}"
CMD_END_PROC="${PYTHON_RUN} -m Common.Analysis.EndProcess"
CMD_END_PROC_SING="${CMD_EXEC} ${IMG_SINGULARITY} ${PYTHON_RUN} -m Common.Analysis.EndProcess"
SINGULARITY=true
PARAM_MULTIJOB_JOB="-nj" #optional parameter to add to the folder name and job
parallel=false
interpretation=true
multi_job=""
params=${@}

if ! command -v $CMD_QUEUE &> /dev/null; then
  CMD_QUEUE="bash"
fi
if [ $SINGULARITY = true ];then
  cmd_run=${CMD_SING}
  cmd_help=${CMD_SING_HELP}
  test_cmd=${TEST_CMD}
else
  cmd_run="${PYTHON_RUN} ${SIBILA}"
  cmd_help="${PYTHON_RUN} ${SIBILA}"
  test_cmd="python3 -m unittest discover"
fi

while [ ${#} -gt 0 ];do

  if [ "${1}" = "${PARAM_MULTIJOB_JOB}" ]; then
    shift
    if [ "${1}" != "" ];then
      params=`echo ${params}| sed "s/-nj ${1}//g"  `
      multi_job=_"${1}"
    fi
  elif [ "${1}" = "-f" ];then
    shift
    folder=${1}
  elif [ "${1}" = "-d" ];then
    shift
    dataset=$(basename ${1})
    dataset=$(echo ${dataset%.*})
  elif [ "${1}" = "--test" ];then
      ${test_cmd}
      exit
  elif [ "${1}" == "--skip-interpretability" ];then
      shift
      interpretation=false
  elif [ "${1}" = "-h" ];then
    $cmd_help $1
    echo "-nj optional parameter to add to the folder name and job"
    echo "--test Execute test"
    exit
  elif [ "${1}" = "-q" ];then
      parallel=true
      shift
  else
        shift
  fi
done

if [ "${folder}" = "" ];then
    folder=${dataset}
fi
folder="${folder}_$(date +'%F')"

if [ "${folder}" != "" ]; then
    folder=${folder}${multi_job}
    mljob=${PWD}/${folder}/job.sh
    NAME_JOB=$NAME_JOB${multi_job}
    if [ ! -d "${folder}" ]; then
      mkdir ${folder}
    fi
    sh $SCRIPT_QUEUE "${PWD}/${folder}/" ${NAME_JOB} ${TIME} ${CPUS} ${MEM} ${PARTITION} > ${mljob}
    echo "${cmd_run} ${params} -f ${folder}" >> ${mljob}
fi

# send sibila.py job and grab job id
export_var="PARTITION=${PARTITION},TIME=${TIME},MEM=${MEM},SINGULARITY=${SINGULARITY},PYTHON_RUN=${PYTHON_RUN},IMG_SINGULARITY=${IMG_SINGULARITY}"
main_job_id=$(${CMD_QUEUE} --export=${export_var} ${mljob} | cut -d ' ' -f 4)

# make a separate script for each interpretability calculation which will be run once the models are trained
if [ ${parallel} == true ]; then
    endjob=${PWD}/${folder}/end_job.sh
    sh ${SCRIPT_QUEUE} "${PWD}/${folder}/jobs/" "end_job" "4:00:00" "1" "${MEM}" "${PARTITION}" > ${endjob}
    if [ ${SINGULARITY} == true ]; then
        echo "${CMD_END_PROC_SING} ${folder}" >> ${endjob}
    else
        echo "${CMD_END_PROC} ${folder}" >> ${endjob}
    fi

    if [ ${interpretation} == true ]; then
        sh ${SCRIPT_QUEUE} "${PWD}/${folder}/" "SIBILA_INTERPRETABILITY" "${TIME}" "1" "${MEM}" "${PARTITION}" > ${folder}/interpretability.sh
        cat ${PWD}/interpretability.sh >> ${folder}/interpretability.sh
        ${CMD_QUEUE} --export=${export_var} --dependency=afterany:${main_job_id} ${folder}/interpretability.sh "${folder}" "${endjob}"
    else
        export_var="PARTITION=${PARTITION},TIME=${TIME},MEM=${MEM},SINGULARITY=${SINGULARITY},PYTHON_RUN=${PYTHON_RUN},IMG_SINGULARITY=${IMG_SINGULARITY}"
        sbatch --export=${export_var} --dependency=afterany:${main_job_id} ${endjob}
    fi
fi
