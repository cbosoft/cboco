import os
import json
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import cv2

from .annotation import Annotation
from .category import Category
from .image import Image


class Dataset:

    @dataclass
    class Statistics:
        num_images: int
        num_annotations: int
        num_annotations_by_dir: Dict[str, int]
        num_annotated_images: int
        num_annotated_images_by_dir: Dict[str, int]
        num_annotations_by_class: Dict[str, int]
        

    def __init__(
            self,
            images: List[Image],
            categories: List[Category],
            annotations: List[Annotation],
            **extra):
        self.images = images
        self.categories = categories
        self.annotations = annotations
        self.extra = extra
    
    @classmethod
    def empty(cls, categories: List[Category], **extra):
        return cls([], categories, [], **extra)
    
    def add_image(self, image: Image):
        image.set_id(len(self.images))
        self.images.append(image)
        return image
    
    @classmethod
    def from_json(cls, fn: str):
        with open(fn) as f:
            data = json.load(f)

        images = [Image(**im) for im in data['images']]
        images_by_id = {image.id: image for image in images}
        categories = [Category(**cat) for cat in data['categories']]
        annotations = [Annotation(**ann, image=images_by_id[ann['image_id']]) for ann in data['annotations']]
        
        return cls(
            images=images,
            categories=categories,
            annotations=annotations,
            **{k: v for k, v in data.items() if k not in {'images', 'categories', 'annotations'}}
        )
    
    def to_dict(self) -> dict:
        return dict(
            images=[im.to_dict() for im in self.images],
            categories=[cat.to_dict() for cat in self.categories],
            annotations=[ann.to_dict() for ann in self.annotations],
            **self.extra
        )
    
    def to_json(self, fn):
        with open(fn, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def collect_statistics(self) -> Statistics:
        num_images = len(self.images)
        num_annotations = len(self.annotations)
        num_annotated_images = 0
        num_annotated_images_by_dir = {}
        num_annotations_by_dir = defaultdict(int)
        num_annotations_by_class = defaultdict(int)
        categories = {
            cat.id: cat.name
            for cat in self.categories
        }
        for image in self.images:
            d = os.path.dirname(image.file_name)
            n = len(image.annotations)
            num_annotations_by_dir[d] += n
            if d not in num_annotated_images_by_dir:
                num_annotated_images_by_dir[d] = 0
            if n:
                num_annotated_images += 1
                num_annotated_images_by_dir[d] += 1
            
            for ann in image.annotations:
                num_annotations_by_class[categories[ann.category_id]] += 1
        
        return self.Statistics(
            num_images,
            num_annotations,
            num_annotations_by_dir,
            num_annotated_images,
            num_annotated_images_by_dir,
            num_annotations_by_class,
        )
