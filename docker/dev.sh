COCO_DATASET=$1
CMU_DATASET=$2
PWD=$(pwd)
RUN_FOLDER="runs/"
SRC_FOLDER="src/"

mkdir $RUN_FOLDER
docker run --gpus all\
        -p 6006:6006\
        -v $COCO_DATASET:/workspace/dataset/coco\
        -v $CMU_DATASET:/workspace/dataset/panoptic\
        -v $PWD/$RUN_FOLDER:/workspace/runs\
        -v $PWD/$SRC_FOLDER:/workspace/src\
        -v /etc/localtime:/etc/localtime:ro\
        --rm -it weiwang/master-thesis:dev /bin/bash