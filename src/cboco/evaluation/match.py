from typing import List, Dict, Tuple, Optional

from ..dataset import Annotation


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