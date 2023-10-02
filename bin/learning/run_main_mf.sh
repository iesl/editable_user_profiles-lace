#!/usr/bin/env bash
# Parse command line args.
while getopts ":a:e:d:s:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        s) suffix=$OPTARG ;;
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

# Source hyperparameters.
data_path="$CUR_PROJ_DIR/datasets_raw/${dataset}/"
run_time=`date '+%Y_%m_%d-%H_%M_%S'`
run_name="${experiment}-${run_time}-${suffix}"
run_path="$CUR_PROJ_DIR/datasets_raw/${dataset}/${experiment}/${run_name}"

# Train the model.
if [[ $action == 'train_model' ]]; then
    log_file="$run_path/train_run_log.txt"
    mkdir -p "$run_path"
    # Base command line call for all models.
    cmd="python3 -W ignore -um src.learning.main_mfbpr  train_predict_mf_model \
            --model_name $experiment \
            --data_path $data_path \
            --dataset $dataset \
            --run_path $run_path"
    eval $cmd 2>&1 | tee -a ${log_file}
fi