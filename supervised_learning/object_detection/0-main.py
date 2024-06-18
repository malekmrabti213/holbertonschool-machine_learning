#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('0-yolo').Yolo

    # import cv2
    # print(cv2.__version__)
    # import pkg_resources
    # installed_packages = pkg_resources.working_set
    # for package in installed_packages:
    #     print(f"{package.key}=={package.version}")
    
    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    yolo.model.summary()
    print('Class names:', yolo.class_names)
    print('Class threshold:', yolo.class_t)
    print('NMS threshold:', yolo.nms_t)
    print('Anchor boxes:', yolo.anchors)