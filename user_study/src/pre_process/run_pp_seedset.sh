#!/usr/bin/env bash
# Parse command line args.
while getopts ":a:u:m:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        u) username=$OPTARG ;;
        m) model_name=$OPTARG ;;
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
if [[ "$action" == '' ]] || [[ "$username" == '' ]]; then
    echo "Must specify action (-a)"
    echo "Must specify username (-u)"
    exit 1
fi

script_name="pre_proc_seedset"
log_dir="$CUR_PROJ_DIR/logs/lace_user_study/pre_process"
mkdir -p "$log_dir"

data_path="$CUR_PROJ_DIR/datasets_raw/s2orccompsci/lace_user_study"

if [[ $action == 'user_preproc' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${username}_logs.txt"
    # Build seed document embeddings for itemknn seed set.
    cmd0="python3 -um src.pre_process.$script_name  build_seed_reps \
                --data_path $data_path --username $username --model_name cospecter --condition otter"
    eval $cmd0 2>&1 | tee ${log_file}
    # Build seed document embeddings for lace seed set since doc embeddings are used for scalable first stage retrieval.
    cmd1="python3 -um src.pre_process.$script_name  build_seed_reps \
                --data_path $data_path --username $username --model_name cospecter --condition maple"
    eval $cmd1 2>&1 | tee ${log_file}
    # Build seed sentence embeddings for lace seed set for LACE reranking.
    cmd2="python3 -um src.pre_process.$script_name  build_seed_reps \
                --data_path $data_path --username $username --model_name miswordbienc --condition maple"
    eval $cmd2 2>&1 | tee ${log_file}
    # Build the keyphrase profile for LACE.
    cmd3="python3 -um src.pre_process.$script_name get_seed_paper_kps \
                --data_path $data_path --username $username --condition maple"
    eval $cmd3 2>&1 | tee -a ${log_file}
    cmd4="python3 -um src.pre_process.$script_name aggregate_user_kps \
                --data_path $data_path --username $username --condition maple"
    eval $cmd4 2>&1 | tee -a ${log_file}
fi


