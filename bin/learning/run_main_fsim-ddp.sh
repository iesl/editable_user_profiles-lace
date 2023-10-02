#!/usr/bin/env bash
# Parse command line args.
use_untrained=false
while getopts ":a:e:d:s:r:g:n" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        s) suffix=$OPTARG ;;
        r) run_name=$OPTARG ;;
        g) num_gpus=$OPTARG ;;
         # If command line switch active then don't shuffle the data.
        n) no_shuffle=true ;;
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
    echo "Must specify action (-a):"
    echo "Must specify experiment (-e):"
    echo "Must specify dataset (-d):"
    echo "Must a meaningful suffix to add to the run directory (-s)."
    exit 1
fi
if [[ "$action" == 'run_saved' ]] && [[ "$run_name" == '' ]]; then
    echo "Must specify dir name of trained model (-r)."
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
warm_start=$(grep -Po '"warm_start":\s*\K[a-z]*' "$config_path")
if [[ $warm_start == 'true' ]];then
  proc_data_path="$CUR_PROJ_DIR/datasets_proc/${dataset}/${model_name}/warm_start"
else
  proc_data_path="$CUR_PROJ_DIR/datasets_proc/${dataset}/${model_name}/cold_start"
fi

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
    else
        train_basename="train"
    fi
    train_file="$proc_data_path/$train_basename.jsonl"
    shuffled_data_path="$run_path/shuffled_data"
    mkdir -p "$shuffled_data_path"
    # Create a subset of training examples.
    temp_train="$shuffled_data_path/$train_basename-subset.jsonl"
    # Grep some  of the necessary configs from  the json config.
    train_size=$(grep -Po '"train_size":\s*\K([0-9]*)' "$config_path")
    epochs=$(grep -Po '"num_epochs":\s*\K([0-9]*)' "$config_path")
    if [[ "$no_shuffle" != true ]] ; then
        head -n "$train_size" "$train_file" > "$temp_train"
        train_file="$temp_train"
        for (( i=0; i<$epochs; i+=1 )); do
            randomseed=$RANDOM
            fname="$shuffled_data_path/$train_basename-$i.jsonl"
            # Perform integer division so each remainder lines are excluded.
            multigpu_split_size=$(($train_size/$num_gpus))
            shuf --random-source=<(get_seeded_random $randomseed) "$train_file" --output="$fname"
            # Create as splits of the train file for each process to consume. Each much be equal sized.
            # Start numeric suffixes with 0 since rank starts at 0.
            # The split file also makes an extra file (numgpus+1) with the remainder of
            # $train_size/$num_gpus which is harmless for now.
            split --suffix-length=1 --numeric-suffixes=0 --additional-suffix="-$i.jsonl" -l $multigpu_split_size "$fname" "$shuffled_data_path/$train_basename-"
            echo "Created: $fname"
        done
        rm $temp_train
    fi
fi

script_name="main_fsim"
source_path="$CUR_PROJ_DIR/experiments/src/learning"

# Train the model.
if [[ $action == 'train_model' ]]; then
    log_file="$run_path/train_run_log.txt"
    mkdir -p "$run_path"
    # Base command line call for all models.
    cmd="python3 -W ignore -um src.learning.$script_name  train_model \
            --model_name $model_name \
            --data_path $proc_data_path \
            --dataset $dataset \
            --run_path $run_path \
            --config_path $config_path \
            --num_gpus $num_gpus"
    eval $cmd 2>&1 | tee -a ${log_file}
    rm -r "$shuffled_data_path"
fi