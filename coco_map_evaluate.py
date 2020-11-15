import argparse
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description="Evaluate mean average precision by json")
    parser.add_argument("ground_truth", type=str,
                        help="JSON file in COCO format with ground truth labels")
    parser.add_argument("predictions", type=str,
                        help="JSON file in COCO format with predictions")
    args = parser.parse_args()

    gt_mscoco_path = Path(args.ground_truth)
    predict_mscoco_path = Path(args.predictions)

    cocoGt = COCO(gt_mscoco_path.as_posix())
    cocoDt = cocoGt.loadRes(predict_mscoco_path.as_posix())
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    main()
