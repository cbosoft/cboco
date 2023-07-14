import itertools
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import torch

from ..dataset import Dataset, Annotation


def get_datasets_intersection(a: Dataset, b: Dataset):
    assert len(a.categories) == len(b.categories), f'{a.categories} != {b.categories}'

    a_images = set(a.images)
    b_images = set(b.images)
    # ensure no images lost due to hash collision
    assert len(a_images) == len(a.images), 'Hash collision in A!'
    assert len(b_images) == len(b.images), 'Hash collision in A!'

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


def match_pred_to_truth(
        true_annotation: Annotation,
        predicted_annotations: List[Annotation],
        ious: Dict[Tuple[int, int], float],
        iou_thresh: float,
        class_agnostic: bool,
) -> Optional[Annotation]:
    matched = sorted([
        pred
        for pred in predicted_annotations
        if (pred.image_id == true_annotation.image_id) and (ious[true_annotation.id, pred.id] > iou_thresh) and (class_agnostic or (true_annotation.category_id == pred.category_id)) 
    ], key=lambda p: -ious[true_annotation.id, p.id])
    if matched:
        return matched[0]
    return None


def match_all_preds_to_truth(
        true_annotations: List[Annotation],
        predicted_annotations: List[Annotation],
        ious: Dict[Tuple[int, int], float],
        iou_thresh: float,
        class_agnostic: bool,
) -> int:
    # reset annotations
    for ann in predicted_annotations:
        ann.is_tp = False
        ann.relevant_iou = 0.0

    tp = 0
    # match up ground truth to predictions
    for true in true_annotations:
        matched = match_pred_to_truth(true, predicted_annotations, ious, iou_thresh, class_agnostic)
        if matched is not None:
            tp += 1
            matched.is_tp = True
            matched.relevant_iou = ious[true.id, matched.id]
    return tp


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
        metrics[f'AP_{tname}'] = calculate_AP(pann, sort_by_iou, gtp)
    
    if len(iou_thresh) > 1:
        metrics['mAP'] = np.mean([v for k, v in metrics.items() if 'AP' in k])
        metrics['mF1'] = np.mean([v for k, v in metrics.items() if 'F1' in k])
    return metrics