#!/bin/bash
echo "Insert robot to frames"

INPUT_FOLDER="/home/david/Datasets/pedestrian detection dataset/"
OUTPUT_FOLDER="/home/david/Datasets/pedestrian detection dataset robot/"


#python robot_augment.py "${INPUT_FOLDER}/backdoor/input" "${OUTPUT_FOLDER}/backdoor/input" --x=0 --y=50 --scale=0.7
#python robot_augment.py "${INPUT_FOLDER}/busStation/input" "${OUTPUT_FOLDER}/busStation/input" --x=200 --y=50 --scale=0.7
#python robot_augment.py "${INPUT_FOLDER}/copyMachine/input" "${OUTPUT_FOLDER}/copyMachine/input" --x=0 --y=100 --scale=0.9
#python robot_augment.py "${INPUT_FOLDER}/cubicle/input" "${OUTPUT_FOLDER}/cubicle/input" --x=190 --y=50 --scale=0.7
#python robot_augment.py "${INPUT_FOLDER}/office/input" "${OUTPUT_FOLDER}/office/input" --x=190 --y=50 --scale=0.7
#python robot_augment.py "${INPUT_FOLDER}/pedestrians/input" "${OUTPUT_FOLDER}/pedestrians/input" --x=0 --y=100 --scale=0.5
#python robot_augment.py "${INPUT_FOLDER}/peopleInShade/input" "${OUTPUT_FOLDER}/peopleInShade/input" --x=0 --y=0 --scale=0.5
#python robot_augment.py "${INPUT_FOLDER}/PETS2006/input" "${OUTPUT_FOLDER}/PETS2006/input" --x=500 --y=130 --scale=0.5
#python robot_augment.py "${INPUT_FOLDER}/skating/input" "${OUTPUT_FOLDER}/skating/input" --x=0 --y=0 --scale=0.5
python robot_augment.py "${INPUT_FOLDER}/sofa/input" "${OUTPUT_FOLDER}/sofa/input" --x=0 --y=0 --scale=0.7
