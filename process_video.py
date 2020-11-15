import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import struct

import cv2
import numpy as np
from tqdm import tqdm

from amphibiandetector_baseline import AmphibianDetectorSSDBaseline
from amphibiandetector_ssd import AmphibianDetectorSSD


def process_video(frames_filenames: List[Path],
                  detector: AmphibianDetectorSSD,
                  debug_info_folder: Path = None) -> Tuple[List, float]:
    """
    Process video as list of frames with AmphibianDetector
    :param frames_filenames: List of frames for input video
    :param detector: Detector for filter out static frames and detections
    :param debug_info_folder: Folder to save debug images
    :return: List of annotations with each detection; Average time for frame processing
    """
    annotations = []
    if debug_info_folder is not None:
        os.makedirs(str(debug_info_folder), exist_ok=True)

    time_measurements = []
    index = 0
    for file_path in tqdm(frames_filenames):
        file_name = file_path.name
        file_id = file_name.split(".jpg")[0]
        file_id = file_id.split("in")[-1]
        file_id = int(file_id)
        file_id = f"{file_path.parent.parent.name}_{str(file_id)}"

        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        bbox_filtered, scores_filtered, img_dif = detector.process_frame(image)
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        time_measurements.append(elapsed_time)

        if (debug_info_folder is not None) and (img_dif is not None):
            img_dif = cv2.merge((img_dif, img_dif, img_dif))
            #img_dif *= 255
            img_dif = img_dif.astype(np.uint8)
            img_dif = cv2.resize(img_dif, (image.shape[1], image.shape[0]))
            for bbox, score in zip(bbox_filtered, scores_filtered):
                img_dif = cv2.rectangle(img_dif, (bbox[1], bbox[0]),
                                        (bbox[3], bbox[2]), (0, 0, 255), 2)
            dif_filename = debug_info_folder / file_name
            cv2.imwrite(str(dif_filename), img_dif)

        for bbox, score in zip(bbox_filtered, scores_filtered):
            top, left, bottom, right = bbox
            label_data = {"id": index,
                          "image_id": file_id,
                          "category_id": 1,
                          "bbox": [left, top, right - left, bottom - top],
                          "score": int(score * 100)}
            index += 1
            annotations.append(label_data)

    return annotations, np.mean(time_measurements)


def main():
    parser = argparse.ArgumentParser(description="Process video as frames sequence saved in jpg images")
    parser.add_argument("video_folder", type=str,
                        help="Folder with jpg images represented frames of video. File names format: in000001.jpg")
    parser.add_argument("output_file", type=str,
                        help="Path to output json file with detection results")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="Number of GPU for process detector")
    parser.add_argument("--detector_model", type=str, default="./models/ssd_mobilenet_v2_coco.pb",
                        help="Path to pb file with detector model")
    parser.add_argument("--m", type=int, default=6,
                        help="Number of bloc for obtaining feature map")
    parser.add_argument("--detection_threshold", type=float, default=0.0,
                        help="Threshold for filtering bounding boxes by detector confidence")
    parser.add_argument("--motion_threshold", type=float, default=0.2,
                        help="Threshold for filtering bounding boxes by motion score")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="How often update background model")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    background_frames = defaultdict(lambda: 0)
    background_frames["skating"] = 172

    video_folder = Path(args.video_folder)
    output_file = Path(args.output_file)

    frames_filenames = video_folder.glob("in*.jpg")
    frames_filenames = list(frames_filenames)

    # Sorted by ID of frames
    frames_ids = []
    for file_path in frames_filenames:
        file_name = file_path.name
        file_id = file_name.split(".jpg")[0]
        file_id = file_id.split("in")[-1]
        file_id = int(file_id)
        frames_ids.append(file_id)
    frames_filenames = [file_name for _, file_name in sorted(zip(frames_ids, frames_filenames))]

    # Process video and save json with results
    detector = AmphibianDetectorSSD(args.detector_model,
                                    m=args.m,
                                    detection_threshold=args.detection_threshold,
                                    motion_threshold=args.motion_threshold,
                                    alpha=args.alpha)

    #detector = AmphibianDetectorSSDBaseline(args.detector_model,
    #                                        detection_threshold=args.detection_threshold,
    #                                        motion_threshold=args.motion_threshold,
    #                                        alpha=args.alpha)

    folder_name = str(video_folder.parent.name)
    background_frame_name = frames_filenames[background_frames[folder_name]]
    background_frame = cv2.imread(str(background_frame_name))
    background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
    detector.initialize_background_model(background_frame)

    json_labels, average_time = process_video(frames_filenames, detector) #,
                                              #debug_info_folder=output_file.parent / video_folder.parent.name)
    with output_file.open("w") as outfile:
        outfile.write(json.dumps(json_labels))

    with (output_file.with_suffix(".time")).open("wb") as outfile:
        bites = struct.pack('f', average_time)
        outfile.write(bites)
        print(f"average_time: {average_time}")

if __name__ == "__main__":
    main()
