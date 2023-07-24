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
) -> List[Tuple[Annotation, Annotation]]:
    """
    given true and predicted annotated instances
    return list of matched truths and preds

    If a truth has no matching preds, it will not be in returned list.
    """
    matches = []
    # match up ground truth to predictions
    for true in true_annotations:
        matched = match_pred_to_truth(true, predicted_annotations, ious, iou_thresh, class_agnostic)
        if matched is not None:
            matches.append((true, matched))
    return matches