from typing import List, Dict
import json

import numpy as np
import cv2

from .annotation import Annotation
from .category import Category
from .image import Image


class Dataset:

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
            json.dump(self.to_dict(), f)