import os

from cboco.dataset import Dataset
from cboco.evaluation import evaluate_dataset
from cboco.evaluation.intersection import get_datasets_intersection


def test_eval_1():
    true = Dataset.from_json(os.path.join('test_data', 'A.json'))
    preds = Dataset.from_json(os.path.join('test_data', 'B.json'))
    results = evaluate_dataset(
        preds,
        true,
        iou_thresh=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    assert abs(results['mF1'] - 0.625) < 1e-9
    assert abs(results['mAP'] - 0.44619047619047614) < 1e-9


def test_intersection():
    true = Dataset.from_json(os.path.join('test_data', 'A.json'))
    true_images = set([i.base_name for i in true.images])
    preds = Dataset.from_json(os.path.join('test_data', 'B.json'))
    preds_images = set([i.base_name for i in preds.images])
    image_lost = set([preds.images.pop()])
    i_preds, i_true = get_datasets_intersection(preds, true)
    i_true_images = set([i.base_name for i in i_true.images])
    i_preds_images = set([i.base_name for i in i_preds.images])
    assert i_true_images == i_preds_images
    assert i_preds_images.union(image_lost) == preds_images == true_images


def test_eval_2():
    true = Dataset.from_json(os.path.join('test_data', 'A.json'))
    preds = Dataset.from_json(os.path.join('test_data', 'B.json'))
    preds.images.pop()
    results_oneless = evaluate_dataset(
        preds,
        true,
        iou_thresh=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    true.images.pop()
    results_same = evaluate_dataset(
        preds,
        true,
        iou_thresh=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    assert all([results_same[k] == results_oneless[k] for k in results_same.keys()])