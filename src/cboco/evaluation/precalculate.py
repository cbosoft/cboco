from typing import List, Dict, Tuple
import itertools

from tqdm import tqdm

from ..dataset import Annotation


def precalculate_combinatorial_ious(tann: List[Annotation], pann: List[Annotation], method: Annotation.IoUMethod, show_progress: bool) -> Dict[Tuple[int, int], float]:
    combinations = itertools.product(tann, pann)
    n = len(tann)*len(pann)
    if show_progress:
        combinations = tqdm(combinations, total=n)
    ious = {}
    for t, p in combinations:
        if t.image_id == p.image_id:
            ious[t.id, p.id] = t.iou(p, method=method)
    return ious
