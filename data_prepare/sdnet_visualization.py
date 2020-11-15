import argparse
import json
import os
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes from coco formatted json")
    parser.add_argument("json_file", type=str,
                        help="Path to json file with bounding boxes in coco format")
    parser.add_argument("output_folder", type=str,
                        help="Folder for saving resulted images with visualization")
    args = parser.parse_args()

    json_file = Path(args.json_file)
    output_folder = Path(args.output_folder)
    os.makedirs(str(output_folder), exist_ok=True)

    with json_file.open() as file:
        coco_data = json.load(file)
        for image_info in coco_data["images"]:
            filename = image_info["filename"]
            image = cv2.imread(filename)

            for label_info in coco_data["annotations"]:
                if label_info["image_id"] == image_info["id"]:
                    bbox = label_info["bbox"]
                    cv2.rectangle(image, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

            cv2.imwrite(str(output_folder/Path(filename).name), image)


if __name__ == "__main__":
    main()
