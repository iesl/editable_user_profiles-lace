#!/usr/bin/env bash
# Parse command line args.
while getopts ":a:i:j:u:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        i) author_id_1=$OPTARG ;;
        j) author_id_2=$OPTARG ;;
        u) username=$OPTARG ;;
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

# Generate the seed papers from semantic scholar.
if [[ $action == 'download_author_seed_papers' ]]; then
  data_path="$CUR_PROJ_DIR/lace_user_study/users"
  mkdir -p "$data_path/$username/otter"
  mkdir -p "$data_path/$username/maple"

  cmd="python3 -um src.pre_process.get_s2_userdata $action --author_uname $username --author_s2id $author_id_1
  --data_path $data_path/$username/otter"
  echo $cmd
  eval $cmd

  cmd="python3 -um src.pre_process.get_s2_userdata $action --author_uname $username --author_s2id $author_id_2
  --data_path $data_path/$username/maple"
  echo $cmd
  eval $cmd

elif [[ $action == 'download_seed_paper_refs' ]]; then
  data_path="$CUR_PROJ_DIR/lace_user_study/users"

  cmd="python3 -um src.pre_process.get_s2_userdata $action --username $username --data_path $data_path/$username/otter"
  echo $cmd
  eval $cmd

  cmd="python3 -um src.pre_process.get_s2_userdata $action --username $username --data_path $data_path/$username/maple"
  echo $cmd
  eval $cmd
fi
