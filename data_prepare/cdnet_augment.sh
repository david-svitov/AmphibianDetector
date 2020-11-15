#!/bin/bash
echo "Apply video degradation to CDNet2014"

INPUT_FOLDER="/home/david/Datasets/pedestrian detection dataset robot/"
OUTPUT_FOLDER="/home/david/Datasets/pedestrian detection dataset robot augmented/"

for folder in "${INPUT_FOLDER}"/*/ ; do
    echo "Process folder ${folder}"
    python camera_artifacts_augment.py "${folder}"/"input" "${OUTPUT_FOLDER}"/$(basename "$folder")/"input"
done

