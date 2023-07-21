from typing import Tuple

from ..dataset import Dataset


def get_datasets_intersection(a: Dataset, b: Dataset) -> Tuple[Dataset, Dataset]:
    assert len(a.categories) == len(b.categories), f'{a.categories} != {b.categories}'

    a_images = set(a.images)
    b_images = set(b.images)
    # ensure no images lost due to hash collision
    assert len(a_images) == len(a.images), 'Hash collision in A!'
    assert len(b_images) == len(b.images), 'Hash collision in B!'

    common_images = set.intersection(a_images, b_images)
    
    a_images = sorted([img for img in a_images if img in common_images], key=lambda i: i.base_name)
    b_images = sorted([img for img in b_images if img in common_images], key=lambda i: i.base_name)
    assert len(a_images) == len(b_images) == len(common_images), 'Images lost along the way!'

    a_ann = []
    for i, img in enumerate(a_images, start=1):
        img.set_id(i)
        a_ann.extend(img.annotations)
    b_ann = []
    for i, img in enumerate(b_images, start=1):
        img.set_id(i)
        b_ann.extend(img.annotations)
    
    return (
        Dataset(a_images, a.categories, a_ann, **a.extra),
        Dataset(b_images, b.categories, b_ann, **b.extra),
    )
