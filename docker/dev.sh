COCO_DATASET=$1
CMU_DATASET=$2
LIGHTENING_LOG_DIR=$3
IMG_LOG_DIR=$4
PRETRAIN_DIR=$5
PWD=$(pwd)
SRC_FOLDER="src/"

mkdir $RUN_FOLDER
docker run --gpus all\
        --shm-size=8g\
        -p 6006:6006\
        -v $COCO_DATASET:/workspace/dataset/coco\
        -v $CMU_DATASET:/workspace/dataset/panoptic\
        -v $LIGHTENING_LOG_DIR:/workspace/lightening_logs\
        -v $IMG_LOG_DIR:/workspace/images\
        -v $PRETRAIN_DIR:/workspace/pretrain\
        -v $PWD/$SRC_FOLDER:/workspace/src\
        -v /etc/localtime:/etc/localtime:ro\
        --rm -it weiwang/master-thesis:dev / 