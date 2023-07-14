from typing import List, Dict, Tuple
import itertools

from tqdm import tqdm

from ..dataset import Annotation


def precalculate_combinatorial_ious(a: List[Annotation], b: List[Annotation], method: Annotation.IoUMethod, show_progress: bool) -> Dict[Tuple[int, int], float]:
    combinations = itertools.product(a, b)
    n = len(a)*len(b)
    it = combinations
    if show_progress:
        it = tqdm(it, total=n)
    ious = {}
    for t, p in it:
        if t.image_id == p.image_id:
            ious[t.id, p.id] = t.iou(p, method=method)
    return ious
