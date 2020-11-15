#!/bin/bash
echo "Process videos in CDNet2014"

DATASET_FOLDER="/home/david/Datasets/pedestrian detection dataset robot augmented/"
PREDICTIONS_FOLDER="/home/david/repository/expasoft/amphibiandetector/outputs/"

for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for folder in "${DATASET_FOLDER}"/*/ ; do
        echo "Process folder ${folder}"
        python process_video.py "${folder}"/input \
        "${PREDICTIONS_FOLDER}"/$(basename "$folder").json \
        --gpu=-1 \
        --m=11 \
        --motion_threshold=${i} \
        --detection_threshold=0.0 \
        --alpha=0.9
    done

    python merge_predictions.py "${PREDICTIONS_FOLDER}" predictions.json
    python time_averaging.py "${PREDICTIONS_FOLDER}" > res${i}.txt

    python coco_map_evaluate.py \
    '/home/david/Datasets/pedestrian detection dataset robot augmented/coco_format.json' \
    ./predictions.json >> res${i}.txt
done
