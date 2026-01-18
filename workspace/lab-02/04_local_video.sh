#!/bin/bash

usage()
{
cat << EOF
USAGE: `basename $0` [options]
    -d  dump file
EOF
}

while getopts "d:h" OPTION
do
    case $OPTION in
        d) path_dump=$OPTARG ;;
        h) usage; exit 1 ;;
        *) usage; exit 1 ;;
    esac
done

if [ -z "$path_dump" ]
then
    usage
    exit 1
fi

python -m gfootball.dump_to_video \
    --trace_file ${path_dump}

python -m gfootball.replay \
    --trace_file ${path_dump}
