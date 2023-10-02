#!/usr/bin/env bash
# Parse command line args.
use_log_file=false
caching_scorer=false
termm_rerank=false
model_version='cur_best'
while getopts ":a:e:r:d:t:v:s:b:lcm" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        t) train_dataset=$OPTARG ;;
        # Trained model basepath name.
        r) run_name=$OPTARG ;;
        v) model_version=$OPTARG ;;
        s) train_suffix=$OPTARG ;;
        # {'big', 'bids', 'assigns'}
        b) ann_suffix=$OPTARG ;;
        # Command line switch to use a log file and not print to stdout.
        l) use_log_file=true ;;
        # Switch to say to use the caching scorer or to generate reps and then score.
        c) caching_scorer=true ;;
        # Run on the test set with larger #interaction users.
        m) termm_rerank=true ;;
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

source_path="$CUR_PROJ_DIR/experiments/src/pre_process"

if [[ "$action" == 'rank_pool' ]]; then
    script_name="pp_gen_nearest"
    root_path="$CUR_PROJ_DIR/datasets_raw/${dataset}/"
    model_base_path="$CUR_PROJ_DIR/model_runs/${train_dataset}"
    log_file="${root_path}/${experiment}/${script_name}-${action}-${train_dataset}-${dataset}-${experiment}_logs.txt"
    cmd="python3 -um src.pre_process.$script_name $action \
      --root_path $root_path --dataset $dataset --rep_type $experiment"
    if [[ $run_name != '' ]]; then
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
elif [[ "$action" == 'rank_tag_pool' ]]; then
    script_name="pp_gen_profilekps"
    root_path="$CUR_PROJ_DIR/datasets_raw/${dataset}/"
#    model_base_path="$CUR_PROJ_DIR/model_runs/${train_dataset}"
    log_file="${root_path}/${experiment}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
    cmd="python3 -um src.pre_process.$script_name $action \
      --root_path $root_path --dataset $dataset --rep_type $experiment"
    if [[ $run_name != '' ]]; then
#        model_path="${model_base_path}/${experiment}/${run_name}"
        log_dir="${root_path}/${experiment}/${run_name}"
        mkdir -p "$log_dir"
        # Copy this over so that generating nearest docs have access to it to compute scores.
        # Do it in update mode only though.
#        cp -u "$model_path/run_info.json" "$log_dir"
        log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
        cmd="$cmd --run_name $run_name"
    else
        log_dir="${root_path}/${experiment}/"
        mkdir -p "$log_dir"
        log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
    fi
    if [[ $use_log_file == true ]]; then
      cmd="$cmd --log_fname $log_file"
    fi
    if [[ $termm_rerank == true ]]; then
      cmd="$cmd --termm_rerank"
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