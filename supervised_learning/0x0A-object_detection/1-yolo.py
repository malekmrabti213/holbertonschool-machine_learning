#!/usr/bin/env python3

import numpy as np
import tensorflow.keras as K

class Yolo:
    """Class to perform the Yolo algorithm on image data"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initializes the Yolo class"""
        self.class_t = class_t # class score threshold
        self.nms_t = nms_t # non max suppression threshold
        self.model = K.models.load_model(model_path) # keras darknet model
        self.anchors = anchors # anchor boxes
        with open(classes_path) as f:
            self.class_names = [class_name.strip() for class_name in f.readlines()]

    def process_outputs(self, outputs, image_size):
        """processes the outputs of the model"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            g_h, g_w = output.shape[:2]
            
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_confidence = np.expand_dims(1/(1 + np.exp(-output[..., 4])), axis=-1)
            box_class_prob = 1/(1 + np.exp(-output[..., 5:]))

            b_wh = anchors * np.exp(t_wh)
            b_wh = b_wh / self.model.inputs[0].shape.as_list()[1:3]
            grid = np.tile(np.indices((g_w, g_h)).T, anchors.shape[0]).reshape((g_h, g_w) + anchors.shape)
            b_xy = (1/(1 + np.exp(-t_xy)) + grid) / [g_w, g_h]
            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box = box * np.tile(np.flip(image_size, axis=0), 2)
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs