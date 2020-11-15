import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge all json files with COCO annotations in subfolders")
    parser.add_argument("folder", type=str,
                        help="Parent folder for search json files")
    parser.add_argument("output_file", type=str,
                        help="Path to output merged json file")
    args = parser.parse_args()

    folder = Path(args.folder)
    output_file = Path(args.output_file)

    json_labels = {"images": [],
                   "annotations": [],
                   "categories": [{"supercategory": "pedestrian",
                                   "id": 1,
                                   "name": "pedestrian"}]
                   }

    json_files = folder.rglob("*/*.json")
    for json_file in json_files:
        with json_file.open() as file:
            coco_data = json.load(file)
            json_labels["images"] += coco_data["images"]
            json_labels["annotations"] += coco_data["annotations"]

    for i in range(len(json_labels["annotations"])):
        json_labels["annotations"][i]["id"] = i

    with output_file.open("w") as outfile:
        outfile.write(json.dumps(json_labels))

if __name__ == "__main__":
    main()
