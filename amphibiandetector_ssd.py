from typing import List
from typing import Tuple
from typing import Dict

import numpy as np
import tensorflow as tf


class AmphibianDetectorSSD:
    """
    Implementation of AmphibianDetector based on SSD
    """
    
    def __init__(self, 
                 path_to_ckpt: str,
                 m: int = 6,
                 detection_threshold: float = 0.3,
                 motion_threshold: float = 0.2,
                 alpha: float = 0.9):
        """
        Initialize AmphibianDetector based on chosen SSD model
        :param path_to_ckpt: Path to pb file with friezed SSD model
        :param m: Number of block in feature extractor for feature map calculation
        :param detection_threshold: Threshold for detector confidences
        :param motion_threshold: Threshold for motion activity level
        :param alpha: How often update background model
        """
        
        self.story_features = None

        self.alpha = alpha
        self.detection_threshold = detection_threshold
        self.motion_threshold = motion_threshold
        
        detection_graph = tf.Graph()
        
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with tf.gfile.GFile(path_to_ckpt, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
                
            feature_layer_in = f"FeatureExtractor/MobilenetV2/expanded_conv_{str(m+1)}/input:0"
            feature_layer_out = f"FeatureExtractor/MobilenetV2/expanded_conv_{str(m)}/output:0"

            self.motion_features_in = self.sess.graph.get_tensor_by_name("image_tensor:0")
            self.detector_in = self.sess.graph.get_tensor_by_name(feature_layer_in)
            self.postprocessor_in = self.sess.graph.get_tensor_by_name("Postprocessor/ToFloat:0")

            self.motion_features_out = {"features": 
                                        self.sess.graph.get_tensor_by_name(feature_layer_out),
                                        "preprocessor":
                                            self.sess.graph.get_tensor_by_name(
                                                "Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0")
                                        }
            self.detection_out = {"detection_boxes": self.sess.graph.get_tensor_by_name("detection_boxes:0"),
                                  "detection_scores": self.sess.graph.get_tensor_by_name("detection_scores:0")}

    def _predict_motion_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Inference first part of CNN for obtain feature map
        :param frame: Input image in SSD input format
        :return: Values for feature map tensor
        """
        image_batch = np.expand_dims(frame, axis=0)
        output_dict = self.sess.run(self.motion_features_out,
                                    feed_dict={self.motion_features_in: image_batch})

        return output_dict

    def _detect_objects(self,
                        features: Dict) -> Tuple[np.ndarray,
                                                 np.ndarray]:
        """
        Finish SSD inference by feature map
        :param features: Feature map from intermediate layer
        :return: Coordinates of detected objects and confidence scores
        """

        output_dict = self.sess.run(self.detection_out,
                                    feed_dict={self.detector_in: features["features"],
                                               self.postprocessor_in: features["preprocessor"]})

        detection_boxes = output_dict["detection_boxes"][0]
        detection_scores = output_dict["detection_scores"][0]

        return detection_boxes, detection_scores

    @staticmethod
    def _cos_distance(feature_map_1: np.ndarray,
                      feature_map_2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine distance between vectors in two features maps
        :param feature_map_1: First feature map
        :param feature_map_2: Second feature map
        :return: Matrix with cosine distances
        """
        norm_1 = np.linalg.norm(feature_map_1, axis=-1)
        norm_2 = np.linalg.norm(feature_map_2, axis=-1)
        feature_map_1 = feature_map_1 / np.expand_dims(norm_1, axis=-1)
        feature_map_2 = feature_map_2 / np.expand_dims(norm_2, axis=-1)
        dist = np.sum(feature_map_1 * feature_map_2, axis=-1)
        return 1 - np.maximum(dist, 0)

    def reset(self):
        """
        Reset background model
        :return:
        """
        self.story_features = None
        
    def initialize_background_model(self,
                                    img: np.ndarray):
        features = self._predict_motion_features(img)
        self.story_features = features["features"][0]

    def process_frame(self, 
                      img: np.ndarray) -> Tuple[List[List[int]],
                                                List[float],
                                                np.ndarray]:
        """
        Process one frame of video with AmphibianDetector
        :param img: Frame of video in format for SSD input
        :return: Bounding boxes, scores, and motion detection map
        """
        bbox_filtered, scores_filtered = [], []
        img_dif = None
        
        features = self._predict_motion_features(img)
        
        if self.story_features is not None:
            img_dif = AmphibianDetectorSSD._cos_distance(features["features"][0], self.story_features)
            if img_dif.max() < self.motion_threshold:
                self.story_features = self.story_features * self.alpha + features["features"][0] * (1 - self.alpha)
        else:
            self.story_features = features["features"][0]
        
        if img_dif is not None and img_dif.max() >= self.motion_threshold:
            detection_boxes, detection_scores = self._detect_objects(features)
            
            for bbox, score in zip(detection_boxes, detection_scores):
                if score < self.detection_threshold:
                    continue
                inner_dif = img_dif[int(bbox[0]*img_dif.shape[0]): int(bbox[2]*img_dif.shape[0]),
                                    int(bbox[1]*img_dif.shape[1]): int(bbox[3]*img_dif.shape[1])]

                if inner_dif.mean() >= self.motion_threshold:
                    postproc_bbox = [int(bbox[0]*img.shape[0]), int(bbox[1]*img.shape[1]),
                                     int(bbox[2]*img.shape[0]), int(bbox[3]*img.shape[1])]
                    bbox_filtered.append(postproc_bbox)
                    scores_filtered.append(score)

        return bbox_filtered, scores_filtered, img_dif
