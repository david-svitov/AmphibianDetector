import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge all json predictions for all videos")
    parser.add_argument("folder", type=str,
                        help="Parent folder for search json files")
    parser.add_argument("output_file", type=str,
                        help="File with bounding boxes in COCO format")
    args = parser.parse_args()

    folder = Path(args.folder)
    output_file = Path(args.output_file)

    annotations = []

    json_files = folder.glob("*.json")
    for json_file in json_files:
        with json_file.open() as file:
            coco_data = json.load(file)
            annotations += coco_data

    for i in range(len(annotations)):
        annotations[i]["id"] = i

    with output_file.open("w") as outfile:
        outfile.write(json.dumps(annotations))


if __name__ == "__main__":
    main()
