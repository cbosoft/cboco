from enum import Enum
from typing import Dict, List

import numpy as np
import cv2

from .image import Image

class Annotation:

    class IoUMethod(Enum):
        Box = 0
        Mask = 1

    def __init__(
            self,
            id: int,
            image_id: int,
            segmentation: List[List[int]],
            category_id: int,
            image: Image,
            bbox: List[int],
            score: float = None,
            iscrowd=0,
            **extra):
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.bbox = bbox
        self.score = score
        self.iscrowd = iscrowd
        if iscrowd:
            raise NotImplementedError
        self.extra = extra

        if image is not None:
            seg = np.array(segmentation)
            self.contour = seg.reshape(-1, 1, 2).astype(np.int32)

            x1 = int(np.min(self.contour[..., 0]))
            x2 = int(np.max(self.contour[..., 0]))
            y1 = int(np.min(self.contour[..., 1]))
            y2 = int(np.max(self.contour[..., 1]))

            self.bbox = x1, y1, x2, y2
            w, h = image.width, image.height
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [self.contour], -1, 1, -1)
            self.mask = mask.astype(bool)

            image.annotations.append(self)
        else:
            self.mask = None
            self.contour = None

        self.is_tp = False
        self.relevant_iou = 0.0
    
    def seg_iou(self, other: "Annotation"):
        i = np.sum(self.mask & other.mask)
        u = np.sum(self.mask | other.mask)
        return float(i) / float(u)
    
    def box_iou(self, other: "Annotation"):
        """https://stackoverflow.com/a/42874377"""
        a_x1, a_y1, a_x2, a_y2 = self.bbox
        assert a_x1 < a_x2
        assert a_y1 < a_y2
        a_w, a_h = a_x2 - a_x1, a_y2 - a_y1
        b_x1, b_y1, b_x2, b_y2 = other.bbox
        assert b_x1 < b_x2
        assert b_y1 < b_y2
        b_w, b_h = b_x2 - b_x1, b_y2 - b_y1

        # determine the coordinates of the intersection rectangle
        x_left = max(a_x1, b_x1)
        y_bottom = max(a_y1, b_y1)
        x_right = min(a_x2, b_x2)
        y_top = min(a_y2, b_y2)

        if x_right < x_left or y_bottom > y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_top - y_bottom)

        # compute the area of both AABBs
        a_area = a_w * a_h
        b_area = b_w * b_h

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(a_area + b_area - intersection_area)
        assert 0.0 <= iou <= 1.0
        return iou 
    
    def iou(self, other: "Annotation", method=IoUMethod.Box):
        if not isinstance(method, self.IoUMethod):
            raise ValueError(f'Unknown IoU method "{method}", expected instance of {self.IoUMethod}.')
        if method == self.IoUMethod.Box:
            return self.box_iou(other)
        elif method == self.IoUMethod.Mask:
            return self.seg_iou(other)
        else:
            raise ValueError(f'Unknown IoU method "{method}".')
    
    def to_dict(self) -> dict:
        return dict(
            id=self.id,
            image_id=self.image_id,
            category_id=self.category_id,
            segmentation=self.segmentation,
            iscrowd=self.iscrowd,
            **self.extra
        )