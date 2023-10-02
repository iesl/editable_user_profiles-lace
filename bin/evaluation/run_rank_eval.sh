#!/usr/bin/env bash
# Parse command line args.
warm_start=false
while getopts ":a:d:e:r:c:w" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        d) dataset=$OPTARG ;;
        e) experiment=$OPTARG ;;
        r) run_name=$OPTARG ;;
        c) comet_exp_key=$OPTARG ;;
        w) warm_start=true ;;
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

script_name="ranking_eval"
if [[ "$action" == 'eval_pool_ranking' ]]; then
    data_path="$CUR_PROJ_DIR/datasets_raw/$dataset"
    run_path="$CUR_PROJ_DIR/datasets_raw/$dataset"
    log_file="${run_path}/${script_name}-${action}-${dataset}-${experiment}-eval_logs.txt"
    cmd="python3 -um src.evaluation.ranking_eval $action \
    --data_path $data_path --experiment $experiment --dataset $dataset"
    if [[ $run_name != '' ]]; then
        run_path="$data_path/$experiment/$run_name"
        log_file="${run_path}/${script_name}-${action}-${dataset}-${experiment}-eval_logs.txt"
        cmd="$cmd --run_path $run_path --run_name $run_name"
    fi
    if [[ $comet_exp_key != '' ]]; then
        cmd="$cmd --comet_exp_key $comet_exp_key"
    fi
    if [[ $warm_start == true ]]; then
        cmd="$cmd --warm_start"
    fi
elif [[ "$action" == 'result_signf' ]]; then
    run_path="$CUR_PROJ_DIR/datasets_raw/$dataset/$experiment/$run_name"
    log_file="${run_path}/${script_name}-${action}-${dataset}-${experiment}-eval_logs.txt"
    cmd="python3 -um src.evaluation.ranking_eval $action --dataset $dataset --method1 $experiment --run_name1 $run_name"
fi


echo "$cmd" | tee ${log_file}
eval "$cmd" 2>&1 | tee -a ${log_file}
