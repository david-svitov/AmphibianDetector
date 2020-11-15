#!/bin/bash
echo "Process videos in CDNet2014"

DATASET_FOLDER="/home/user/David/Datasets/CDNet2014/pedestrian detection dataset/"

for folder in "${DATASET_FOLDER}"/*/ ; do
    echo "Process folder ${folder}"
    python cdnet_pedastrian_prepare.py "${folder}" "${folder}"
done

python merge_json.py "${DATASET_FOLDER}" "${DATASET_FOLDER}"/coco_format.json
