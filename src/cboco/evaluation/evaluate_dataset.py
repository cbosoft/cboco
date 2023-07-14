from typing import Dict, List

import numpy as np

from ..dataset import Dataset, Annotation

from .match import match_all_preds_to_truth
from .precalculate import precalculate_combinatorial_ious
from .ap import calculate_AP
from .intersection import get_datasets_intersection


def evaluate_dataset(
        preds: Dataset,
        truth: Dataset,
        iou_method=Annotation.IoUMethod.Box,
        iou_thresh=0.5,
        class_agnostic=False,
        sort_by_iou=False,
        show_progress=True,
) -> Dict[str, float]:
    assert len(preds.categories) == len(truth.categories), f'{preds.categories} != {truth.categories}'

    preds, truth = get_datasets_intersection(preds, truth)
    pann, tann = preds.annotations, truth.annotations

    ious = precalculate_combinatorial_ious(truth.annotations, preds.annotations, iou_method, show_progress)

    should_calc_AP = sort_by_iou or pann[0].score
    
    # ensure IoU thresh is iterable
    try:
        _ = len(iou_thresh)
    except TypeError:
        iou_thresh = [iou_thresh]
    
    gtp = len(tann)
    
    metrics = {}
    for thresh in iou_thresh:
        tname = str(int(thresh*100))
        tp = match_all_preds_to_truth(tann, pann, ious, thresh, class_agnostic)
        fp = len(pann) - tp
        p = tp / (tp + fp)
        r = tp / gtp
        f1 = 2*p*r/(p + r) if tp else 0.0
        metrics[f'P_{tname}'] = p
        metrics[f'R_{tname}'] = r
        metrics[f'F1_{tname}'] = f1

        if should_calc_AP:
            metrics[f'AP_{tname}'] = calculate_AP(pann, sort_by_iou, gtp)
    
    if len(iou_thresh) > 1:
        metrics['mAP'] = np.mean([v for k, v in metrics.items() if 'AP' in k])
        metrics['mF1'] = np.mean([v for k, v in metrics.items() if 'F1' in k])
    return metrics