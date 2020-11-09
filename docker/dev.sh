COCO_DATASET=$1
CMU_DATASET=$2
LIGHTENING_LOG_DIR=$3
IMG_LOG_DIR=$3
PWD=$(pwd)
RUN_FOLDER="runs/"
SRC_FOLDER="src/"

mkdir $RUN_FOLDER
docker run --gpus all\
        --shm-size=8g\
        -p 6006:6006\
        -v $COCO_DATASET:/workspace/dataset/coco\
        -v $CMU_DATASET:/workspace/dataset/panoptic\
        -v $LIGHTENING_LOG_DIR:/workspace/lightening_logs\
        -v $IMG_LOG_DIR:/workspace/images\
        -v $PWD/$SRC_FOLDER:/workspace/src\
        -v /etc/localtime:/etc/localtime:ro\
        --rm -it weiwang/master-thesis:dev /bin/bash