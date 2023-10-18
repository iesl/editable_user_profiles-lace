#!/usr/bin/env bash
# Parse command line args.
use_log_file=false
caching_scorer=false
model_version='cur_best'
while getopts ":a:e:d:t:r:v:s:j:b:lc" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        # Trained model basepath name.
        t) train_dataset=$OPTARG ;;
        r) run_name=$OPTARG ;;
        v) model_version=$OPTARG ;;
        s) train_suffix=$OPTARG ;;
        # Pass a json config to run a model (available on hf) in inference model
        j) config_path=$OPTARG ;;
        # {'big', 'bids', 'assigns'}
        b) ann_suffix=$OPTARG ;;
        # Command line switch to use a log file and not print to stdout.
        l) use_log_file=true ;;
        # Switch to say to use the caching scorer or to generate reps and then score.
        c) caching_scorer=true ;;
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

# $CUR_PROJ_ROOT is a environment variable; manually set outside the script.
log_dir="$CUR_PROJ_DIR/logs/pre_process"
mkdir -p $log_dir

if [[ "$action" == 'rank_pool' ]]; then
    script_name="pp_gen_nearest"
    root_path="$CUR_PROJ_DIR/datasets_raw/${dataset}/"
    cmd="python3 -um src.pre_process.$script_name $action --root_path $root_path --dataset $dataset --rep_type $experiment"
    if [[ $run_name != '' ]]; then
        model_base_path="$CUR_PROJ_DIR/model_runs/${train_dataset}"
        model_path="${model_base_path}/${experiment}/${run_name}"
        log_dir="${root_path}/${experiment}/${run_name}"
        mkdir -p "$log_dir"
        # Copy this over so that generating nearest docs have access to it to compute scores.
        # Do it in update mode only though.
        cp -u "$model_path/run_info.json" "$log_dir"
        if [ -n "$ann_suffix" ]; then  # https://stackoverflow.com/a/3601734/3262406
          log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}-${ann_suffix}_logs.txt"
        else
          log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
        fi
        cmd="$cmd --run_name $run_name --model_path $model_path --model_version $model_version"
    else
      mkdir -p "${root_path}/${experiment}/manual_run"
      log_file="${root_path}/${experiment}/manual_run/${script_name}-${action}-${train_dataset}-${dataset}-${experiment}_logs.txt"
    fi
    if [[ $config_path != '' ]]; then
      cmd="$cmd --config_path $config_path"
    fi
    if [[ $train_suffix != '' ]]; then
      cmd="$cmd --train_suffix $train_suffix"
    fi
    if [[ $use_log_file == true ]]; then
      cmd="$cmd --log_fname $log_file"
    fi
    if [[ $caching_scorer == true ]]; then
      cmd="$cmd --caching_scorer"
    fi
    if [ -n "$ann_suffix" ]; then
      cmd="$cmd --ann_suffix $ann_suffix"
    fi
else
    echo "Unknown action."
    exit 1
fi

if [[ $use_log_file == true ]]; then
  eval $cmd
else
  eval $cmd 2>&1 | tee ${log_file}
fi