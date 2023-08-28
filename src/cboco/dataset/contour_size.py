from typing import Tuple
import cv2
import numpy as np
from imutils import perspective


def midp(pt1, pt2):
    return np.mean([pt1, pt2], axis=0)


def dist(pt1, pt2) -> float:
    return np.sum((pt2-pt1)**2)**0.5


def get_axes(box):
    """
    Get two axis of box (e.g. horiz. and vert.).

    Returns list of tuples of two points defining each axis.
    """
    tl, tr, br, bl = box
    t = midp(tl, tr)
    b = midp(bl, br)
    l = midp(tl, bl)
    r = midp(tr, br)
    return [(t, b), (l, r)]


def size_of_box(box):
    """
    Get size of rectange with corners defined by $box.

    Return (width, length).
    """
    axA, axB = get_axes(box)
    size = dist(*axA), dist(*axB)
    size = tuple(sorted(size))
    return size


def measure_size_of_contour(contour) -> Tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = perspective.order_points(box)
    width, length = size_of_box(box)
    return width, length
