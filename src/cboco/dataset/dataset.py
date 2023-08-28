import os
import re
import json
from typing import List, Dict, Callable, Tuple, Pattern
from dataclasses import dataclass
from collections import defaultdict
import shutil
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import cv2

from .annotation import Annotation
from .category import Category
from .image import Image


FILTER_FUNC = Callable[[List[Image], int], List[Image]]


class Dataset:

    @dataclass
    class Statistics:
        num_images: int
        num_annotations: int
        num_annotations_by_dir: Dict[str, int]
        num_annotated_images: int
        num_annotated_images_by_dir: Dict[str, int]
        num_annotations_by_class: Dict[str, int]
        mean_length: float
        stddev_length: float
        mean_width: float
        stddev_width: float
        mean_aspect_ratio: float
        stddev_aspect_ratio: float
        

    def __init__(
            self,
            images: List[Image],
            categories: List[Category],
            annotations: List[Annotation],
            root='.',
            **extra):
        self.images = images
        self.categories = categories
        self.annotations = annotations
        self.extra = extra

        self.root = root
    
    @classmethod
    def empty(cls, categories: List[Category], **extra) -> "Dataset":
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
            root=os.path.dirname(fn),
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
    
    def to_json(self, fn) -> "Dataset":
        with open(fn, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return self

    @staticmethod
    def scales_from_strs(scales: List[str]) -> List[Tuple[Pattern, float]]:
        rv = []
        for src in scales:
            if ':' not in src:
                pattern_src = '.*'
                scale_src = src
            else:
                pattern_src, scale_src = src.split(':')

            pattern = re.compile('.*'+pattern_src+'.*')

            if '/' in scale_src:
                num, den = [float(v) for v in scale_src.split('/')]
                scale = num/den
            else:
                scale = float(scale_src)

            rv.append((pattern, scale))
        return list(reversed(rv))

    def collect_statistics(self, scales: List[str]) -> Statistics:
        scales = self.scales_from_strs(scales)
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
        lengths = []
        widths = []
        for image in self.images:
            d = os.path.dirname(image.file_name)
            n = len(image.annotations)
            num_annotations_by_dir[d] += n
            if d not in num_annotated_images_by_dir:
                num_annotated_images_by_dir[d] = 0
            if n:
                num_annotated_images += 1
                num_annotated_images_by_dir[d] += 1
            
            scale = None if scales else 1.0
            for p, s in scales:
                if p.match(image.file_name):
                    scale = s
                    break
            assert scale is not None, f'Length scale is set, but no pattern matched image file name "{image.file_name}"!'

            for ann in image.annotations:
                num_annotations_by_class[categories[ann.category_id]] += 1
                w, l = ann.get_width_length(scale)
                widths.append(w)
                lengths.append(l)

        aspect_ratios = np.divide(widths, lengths)

        return self.Statistics(
            num_images,
            num_annotations,
            num_annotations_by_dir,
            num_annotated_images,
            num_annotated_images_by_dir,
            num_annotations_by_class,
            mean_length=np.mean(lengths),
            stddev_length=np.std(lengths),
            mean_width=np.mean(widths),
            stddev_width=np.std(widths),
            mean_aspect_ratio=np.mean(aspect_ratios),
            stddev_aspect_ratio=np.std(aspect_ratios),
        )
    
    def filter_images(self, f: Callable[[Image], bool]) -> "Dataset":
        images = [deepcopy(im) for im in self.images if f(im)]
        annotations = []
        for i, im in enumerate(images, start=1):
            im.set_id(i)
            annotations.extend(im.annotations)
        return Dataset(images, self.categories, annotations, self.root, **self.extra)
    
    @staticmethod
    def random_filter(images: List[Image], count: int) -> List[Image]:
        return list(np.random.choice(images, count))
    
    def subset(self, method: str, by_dir: bool, count: int) -> "Dataset":
        func = None
        if method == 'random':
            func = self.random_filter
        else:
            raise ValueError(f'Unknown subset method {method}.')

        assert func is not None

        if by_dir:
            return self._subset_by_dir(count, func)
        else:
            return self._subset_by_total(count, func)
    
    def _subset_by_dir(self, count: int, filter_func: FILTER_FUNC) -> "Dataset":
        ds = Dataset.empty(self.categories, root=self.root, **self.extra)
        images_by_dir = {}
        for im in self.images:
            im_dir = os.path.dirname(im.file_name)
            if im_dir not in images_by_dir:
                images_by_dir[im_dir] = []
            images_by_dir[im_dir].append(im)
        ds.images = []
        for images in images_by_dir.values():
            images = filter_func(images, count)
            ds.images.extend(images)
        ds.annotations = []
        for i, im in enumerate(ds.images, start=1):
            im.set_id(i)
            ds.annotations.extend(im.annotations)
        return ds

    def _subset_by_total(self, count: int, filter_func: FILTER_FUNC) -> "Dataset":
        ds = Dataset.empty(self.categories, root=self.root, **self.extra)
        ds.images = filter_func(self.images, count)
        ds.annotations = []
        for i, im in enumerate(ds.images, start=1):
            im.set_id(i)
            ds.annotations.extend(im.annotations)
        return ds

    def copy_files(self, dn: str) -> "Dataset":
        if dn == self.root:
            return self
        
        for image in tqdm(self.images, unit='images'):
            src = os.path.join(self.root, image.file_name)
            dest = os.path.join(dn, image.file_name)
            dest_dir = os.path.dirname(dest)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src, dest)
        return self
    
    def union(self, *others: "Dataset", collision_strategy='error') -> "Dataset":
        def update_imset(im: Image, imset: dict) -> Image:
            fn = im.file_name
            if fn in imset:
                if collision_strategy == 'error':
                    raise ValueError(f'Image {fn} present in two or more datasets')
                elif collision_strategy == 'merge':
                    imset[fn].annotations.extend(im.annotations)
                elif collision_strategy == 'preserve':
                    pass
                else:
                    raise ValueError(f'Unknown collision strategy "{collision_strategy}" in union')
            else:
                imset[fn] = im

        image_set = {
            image.file_name: image
            for image in self.images
        }
        for other in others:
            for image in other.images:
                update_imset(image, image_set)

        self.images = list(image_set.values())
        self.annotations = []
        for i, im in enumerate(self.images, start=1):
            im.set_id(i)
            self.annotations.extend(im.annotations)
        return self
        
        
