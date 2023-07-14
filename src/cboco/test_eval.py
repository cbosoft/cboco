import os

from cboco.dataset import Dataset
from cboco.evaluation import evaluate_dataset


def test_eval():
    true = Dataset.from_json(os.path.join('test_data', 'A.json'))
    preds = Dataset.from_json(os.path.join('test_data', 'B.json'))
    results = evaluate_dataset(
        preds,
        true,
        iou_thresh=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    assert abs(results['mF1'] - 0.625) < 1e-9
    assert abs(results['mAP'] - 0.44619047619047614) < 1e-9