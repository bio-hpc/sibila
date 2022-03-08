#!/bin/bash
#
# Parameters job
#
CPUS=1
TIME=24:00:00
NAME_JOB="ML"
MEM=2000M
PROJECT="" #at the moment it is not used
#
#  Constans
#
QUEUE_MANAGER="SLURM"
PYTHON_RUN="python3"
SIBILA="sibila.py"
SCRIPT_QUEUE="Tools/Bash/Queue_manager/${QUEUE_MANAGER}.sh"
CMD_QUEUE="sbatch"
IMG_SINGULARITY="Tools/Singularity/sibila.simg"
TEST_CMD="singularity exec ${IMG_SINGULARITY} python3 -m unittest discover"
CMD_SING="singularity exec ${IMG_SINGULARITY} ${PYTHON_RUN} ${SIBILA}"
CMD_SING_HELP="singularity exec ${IMG_SINGULARITY} ${PYTHON_RUN} ${SIBILA}"
SINGULARITY=true
PARAM_MULTIJOB_JOB="-nj" #optional parameter to add to the folder name and job
CMD_INTERPRETABILITY="${PYTHON_RUN} -m Common.Analysis.Interpretability "
CMD_INTERPRETABILITY_SING="singularity exec ${IMG_SINGULARITY} ${CMD_INTERPRETABILITY} "
CLASS_INTERPRETABILITY="Common/Analysis/Interpretability.py"
CMD_ENDPROCESS="${PYTHON_RUN} -m Common.Analysis.EndProcess "
CMD_ENDPROCESS_SIMG="singularity exec ${IMG_SINGULARITY} ${CMD_ENDPROCESS}"
parallel=false
multi_job=""
params=${@}

if ! command -v $CMD_QUEUE &> /dev/null; then
  CMD_QUEUE="bash"
fi
if [ $SINGULARITY = true ];then
  cmd_run=${CMD_SING}
  cmd_help=${CMD_SING_HELP}
  CMD_INTERPRETABILITY=${CMD_INTERPRETABILITY_SING}
  CMD_ENDPROCESS=${CMD_ENDPROCESS_SIMG}
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
      params=`echo ${params}| sed "s/-nj ${1}//g"  ` #| sed  "s/-nj  $1//g"`
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
    sh $SCRIPT_QUEUE "${PWD}/${folder}/" ${NAME_JOB} ${TIME} ${CPUS} ${MEM} > $mljob
    echo "${cmd_run} ${params} -f ${folder}" >> $mljob
fi

if [ ${parallel} == true ]; then
    methods=$(cat ${CLASS_INTERPRETABILITY} |grep "PARALLEL_METHODS =" |awk -F\= '{print $2}' | tr -d '[],"'  | sed -e "s/'//g" )
    echo -e "#\n#\t Interpretability \n#"                                           >> ${mljob}
    echo -e 'nresults=`ls '${PWD}/${folder}/*_params.pkl' 2> /dev/null |wc -l`'     >> ${mljob}
    echo -e 'if [ ${nresults} -eq 0 ];then exit;fi'                                 >> ${mljob}
    echo -e 'res=`find '${PWD}/${folder}' -empty -name *_params.pkl`'               >> ${mljob}
    echo -e 'if [ "${res}" != "" ];then exit;fi'                                    >> ${mljob}
    echo -e "dependences=afterany"                                                  >> ${mljob}
    echo -e "methods=\"$methods\""                                                  >> ${mljob}
    echo "IFS=' '"                                                                  >> ${mljob}
    echo 'read -ra arr <<< ${methods}'                                              >> ${mljob}
    echo 'for m in ${arr[@]}; do'                                                   >> ${mljob}
    echo -e   "\t for i in  ${folder}/*_params.pkl ; do"                            >> ${mljob}
    echo -e     '\t\t ba=$(basename -- $i})'                                        >> ${mljob}
    echo -e     '\t\t method_ml=`echo ${ba} | cut -d "_" -f 1`'                     >> ${mljob}
    echo -e     "\t\t job=${PWD}/${folder}/"'${method_ml}_${m}_job.sh'              >> ${mljob}
    echo -e     '\t\t name_job=${method_ml}_${m}'                                   >> ${mljob}
    echo -e     "\t\t sh $SCRIPT_QUEUE ${PWD}/${folder}/"'${method_ml}_${m}_ ${name_job}' "${TIME} ${CPUS} ${MEM}"' > $job'  >> ${mljob}
    echo -e     '\t\t echo "'$CMD_INTERPRETABILITY' ${i} ${m}" >> ${job}'           >> ${mljob}
    echo -e     "\t\t j_d=\`$CMD_QUEUE " '${job}`'                                  >> ${mljob}
    echo -e     '\t\t j_d=`echo ${j_d##* }`'                                        >> ${mljob}
    echo -e     '\t\t dependences=${dependences}:${j_d}'                            >> ${mljob}
    # echo -e     '\t\t echo ${dependences}'                                        >> ${mljob}
    # echo -e    '\t\t echo "$CMD_INTERPRETABILITY ${i} ${m} >> $job"'              >> ${mljob}
    echo -e   "\t done"                                                             >> ${mljob}
    echo  "done"                                                                    >> ${mljob}
    echo -e "#\n#\t End Job \n#"                                                    >> ${mljob}
    echo -e     "job=${PWD}/${folder}/end_job.sh"                                   >> ${mljob}
    echo -e     "sh $SCRIPT_QUEUE ${PWD}/${folder}/"'end_job_ end_job' "4:00:00 ${CPUS} ${MEM}"' > $job'  >> ${mljob}
    echo -e    "echo ${CMD_ENDPROCESS_SIMG} ${folder} >> "'${job}'                  >> ${mljob}
    if [ ${CMD_QUEUE} == "bash" ]; then
      echo -e    "${CMD_QUEUE}"' ${job}'                  >> ${mljob}
    else
      echo -e    "${CMD_QUEUE}"' --dependency=${dependences} ${job}'                  >> ${mljob}
    fi
fi


$CMD_QUEUE ${mljob}

