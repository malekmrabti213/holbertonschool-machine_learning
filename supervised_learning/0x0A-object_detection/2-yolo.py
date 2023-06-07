#!/usr/bin/env python3
"""
Initialize Yolo
"""


import numpy as np
import tensorflow.keras as K


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.class_t = class_t
        self.nms_t = nms_t
        self.model = K.models.load_model(model_path)
        self.anchors = anchors

        with open(classes_path, 'r') as f:
            self.class_names = [class_name.strip() for class_name in f.readlines()]

    def process_outputs(self, outputs, image_size):
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, num_anchors = output.shape[:3]

            # Extract the bounding box coordinates and confidence
            bbox_xy = output[..., :2]
            bbox_wh = output[..., 2:4]
            bbox_confidence = np.expand_dims(1 / (1 + np.exp(-output[..., 4])), axis=-1)
            bbox_class_probs = 1 / (1 + np.exp(-output[..., 5:]))

            # Calculate the bounding box coordinates
            anchors = self.anchors[:num_anchors]
            bbox_wh = anchors * np.exp(bbox_wh) / self.model.input.shape[1:3]
            grid_x = np.tile(np.arange(grid_width), grid_height).reshape((grid_height, grid_width, 1, 1))
            grid_y = np.tile(np.arange(grid_height).reshape(-1, 1), grid_width).reshape((grid_height, grid_width, 1, 1))
            bbox_xy = (1 / (1 + np.exp(-bbox_xy)) + np.concatenate((grid_x, grid_y), axis=-1)) / (grid_width, grid_height)
            bbox_xy1 = bbox_xy - bbox_wh / 2
            bbox_xy2 = bbox_xy + bbox_wh / 2
            bbox = np.concatenate((bbox_xy1, bbox_xy2), axis=-1)

            # Scale the bounding box coordinates to the original image size
            image_height, image_width = image_size
            bbox[..., [0, 2]] *= image_width
            bbox[..., [1, 3]] *= image_height

            boxes.append(bbox)
            box_confidences.append(bbox_confidence)
            box_class_probs.append(bbox_class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(boxes, box_confidences, box_class_probs):
            # Calculate box scores
            box_scores = confidence * class_probs

            # Apply class threshold
            box_mask = np.max(box_scores, axis=-1) >= self.class_t
            filtered_boxes.append(box[box_mask])
            box_classes.append(np.argmax(box_scores, axis=-1)[box_mask])
            box_scores.append(np.max(box_scores, axis=-1)[box_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores