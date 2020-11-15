import argparse
import os

import cv2

from amphibiandetector_ssd import AmphibianDetectorSSD

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    parser = argparse.ArgumentParser(description="Process video stream from selected source")
    parser.add_argument("stream", type=str,
                        help="Source of video stream")
    args = parser.parse_args()

    detector = AmphibianDetectorSSD("./models/ssd_mobilenet_v2_coco.pb",
                                    m=11,
                                    detection_threshold=0.4,
                                    motion_threshold=0.4,
                                    alpha=0.9)

    stream = args.stream
    if stream.isdigit():
        stream = int(stream)
    cap = cv2.VideoCapture(stream)

    for _ in range(10):
        ret, background_frame = cap.read()
        assert ret, "Error get frame from stream"
    detector.initialize_background_model(background_frame)

    while(True):
        ret, frame = cap.read()
        assert ret, "Error get frame from stream"

        bbox_filtered, _, _ = detector.process_frame(frame)
        for bbox in bbox_filtered:
            frame = cv2.rectangle(frame, (bbox[1], bbox[0]),
                                         (bbox[3], bbox[2]), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
