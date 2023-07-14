from typing import List

import numpy as np

from ..dataset import Annotation


def calculate_AP(predicted_matched_annotations: List[Annotation], sort_by_iou: bool, gtp: int) -> float:
    ps, rs = [], []
    tp, fp = 0, 0
    for pred in sorted(predicted_matched_annotations, key=lambda p: -p.relevant_iou if sort_by_iou else -p.score):
        if pred.is_tp:
            tp += 1
        else:
            fp += 1
        
        ps.append(tp / (tp + fp))
        rs.append(tp / gtp)
    
    # interpolate precision to be monotonically decreasing
    pinterp = []
    for i in range(len(rs)):
        pinterp.append(max(ps[-i-1:]))
    pinterp = pinterp[::-1]

    # return area under (interpolated) precision-recall curve
    return float(np.trapz(pinterp, rs))