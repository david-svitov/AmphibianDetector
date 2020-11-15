import argparse
import json
from pathlib import Path
from typing import List, Dict

from collections import defaultdict
import imagesize
import pandas as pd


def process_images(image_folder: Path) -> List[Dict]:
    """
    Prepare list of images in COCO format
    :param image_folder: Folder with JPG images with names in format in000001.jpg
    :return: List of images properties
    """
    images = []
    files = image_folder.glob("*.jpg")

    for file_path in files:
        file_name = file_path.name
        file_id = file_name.split(".jpg")[0]
        file_id = file_id.split("in")[-1]
        file_id = int(file_id)
        file_id = f"{file_path.parent.parent.name}_{str(file_id)}"

        width, height = imagesize.get(str(file_path))

        image_data = {"id": file_id,
                      "width": width,
                      "height": height,
                      "filename": str(file_path)}
        images.append(image_data)

    return images


def process_bounding_boxes(ground_truth_file: Path) -> List[Dict]:
    """
    Prepare bounding box annotations in COCO format
    :param ground_truth_file: Path to gt.txt with annotations to images
    :return: List of annotations for images
    """
    index_shifts = defaultdict(lambda: 0)
    index_shifts["cubicle"] = 1108
    folder_name = ground_truth_file.parent.name
    
    annotations = []
    ground_truth_dataframe = pd.read_csv(str(ground_truth_file),
                                         header=None,
                                         sep=" ",
                                         names=["frameID", "0", "x", "y", "width", "height"])
    for index, row in ground_truth_dataframe.iterrows():
        image_id = int(row["frameID"]) + index_shifts[folder_name]
        image_id = f"{folder_name}_{str(image_id)}"
        x = int(row["x"])
        y = int(row["y"])
        width = int(row["width"])
        height = int(row["height"])

        area = width * height
        bbox = [x, y, width, height]

        label_data = {"id": index,
                      "image_id": image_id,
                      "category_id": 1,
                      "bbox": bbox,
                      "area": area,
                      "iscrowd": False}
        annotations.append(label_data)

    return annotations


def main():
    parser = argparse.ArgumentParser(description="Convert CDNet2014 pedestrian to COCO format for validation")
    parser.add_argument("dataset_folder", type=str,
                        help="Folder with pedestrian dataset. With 'input' folder and gt.txt")
    parser.add_argument("output_folder", type=str,
                        help="Folder with output json file")
    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    output_folder = Path(args.output_folder)

    json_labels = {"images": process_images(dataset_folder.joinpath("input")),
                   "annotations": process_bounding_boxes(dataset_folder.joinpath("gt.txt")),
                   "categories": [{"supercategory": "pedestrian",
                                   "id": 1,
                                   "name": "pedestrian"}]
                   }

    with output_folder.joinpath("coco_format.json").open("w") as outfile:
        outfile.write(json.dumps(json_labels))


if __name__ == "__main__":
    main()
