#!/bin/bash
echo "Process videos in CDNet2014"

DATASET_FOLDER="/home/david/Datasets/pedestrian detection dataset robot augmented/backdoor/"
PREDICTIONS_FOLDER="/home/david/repository/expasoft/amphibiandetector/outputs/"


for layer_num in 1 2 3 4 5 6 7 8 9 10 11; do
  for threshold in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "Process folder ${DATASET_FOLDER}"
    python process_video.py "${DATASET_FOLDER}"/input \
    "${PREDICTIONS_FOLDER}"/predictions.json \
    --gpu=-1 \
    --m=${layer_num} \
    --motion_threshold=${threshold} \
    --detection_threshold=0.0 \
    --alpha=0.9

    python time_averaging.py "${PREDICTIONS_FOLDER}" > res_${layer_num}_${threshold}.txt

    python coco_map_evaluate.py \
    "${DATASET_FOLDER}"/coco_format.json \
    "${PREDICTIONS_FOLDER}"/predictions.json >> res_${layer_num}_${threshold}.txt
  done
done
