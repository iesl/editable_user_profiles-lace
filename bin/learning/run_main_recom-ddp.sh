#!/usr/bin/env bash
# Parse command line args.
while getopts ":a:e:d:s:r:g:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        s) suffix=$OPTARG ;;
        # If this is passed in optionally then use it.
        r) run_name=$OPTARG ;;
        g) num_gpus=$OPTARG ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done
# Make sure required arguments are passed.
if [[ "$action" == '' ]] || [[ "$dataset" == '' ]] || [[ "$experiment" == '' ]] || [[ "$suffix" == '' ]]; then
    echo "Must specify action (-a)"
    echo "Must specify experiment (-e)"
    echo "Must specify dataset (-d)"
    echo "Must a meaningful suffix to add to the run directory (-s)."
    exit 1
fi

# Getting a random seed as described in:
# https://www.gnu.org/software/coreutils/manual/html_node/Random-sources.html
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

# Source hyperparameters.
config_path="$CUR_PROJ_DIR/experiments/config/models_config/${dataset}/${experiment}.json"

model_name=$(grep -Po '"model_name":\s*"\K(.*?)(?=")' "$config_path")
run_time=`date '+%Y_%m_%d-%H_%M_%S'`
proc_data_path="$CUR_PROJ_DIR/datasets_proc/${dataset}/${model_name}"

# Create shuffled copies of the dataset; one for each epoch.
if [[ $action == 'train_model' ]]; then
    if [[ $run_name == '' ]]; then
      run_name="${model_name}-${run_time}-${suffix}"
    fi
    run_path="$CUR_PROJ_DIR/model_runs/${dataset}/${model_name}/${run_name}"
    # regex from here: https://stackoverflow.com/q/44524643/3262406
    train_suffix=$(grep -Po '"train_suffix":\s*"\K(.*?)(?=")' "$config_path")
    if [[ $train_suffix != '' ]]; then
        train_basename="train-$train_suffix"
        dev_basename="dev-$train_suffix"
    else
        train_basename="train"
        dev_basename="dev"
    fi
fi

# Train the model.
echo "GPU ids: $CUDA_VISIBLE_DEVICES"

if [[ $action == 'train_model' ]]; then
    log_file="$run_path/train_run_log.txt"
    mkdir -p "$run_path"
    # Base command line call for all models.
    cmd="python3 -W ignore -um src.learning.main_recom  train_model \
            --model_name $model_name \
            --data_path $proc_data_path \
            --dataset $dataset \
            --run_path $run_path \
            --config_path $config_path \
            --num_gpus $num_gpus"
    eval $cmd 2>&1 | tee -a ${log_file}
fi