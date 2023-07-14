from cboco.dataset import Annotation
from cboco.evaluation.match import match_pred_to_truth
from cboco.evaluation.precalculate import precalculate_combinatorial_ious


def test_annot_box_iou_1():
    a = Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )
    b = Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )

    iou = Annotation.box_iou(a, b)

    assert abs(iou - 1.0) < 1e-9


def test_annot_box_iou_2():
    a = Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )
    b = Annotation(
        1, 1, [], 1, None, (12.5, 12.5, 62.5, 62.5), 1.0
    )

    iou = Annotation.box_iou(a, b)

    assert abs(iou - (9./23.)) < 1e-9


def test_annot_match_1():
    a = [Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )]
    b = [Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )]
    ious = precalculate_combinatorial_ious(a, b, Annotation.IoUMethod.Box, show_progress=False)
    rv = match_pred_to_truth(a[0], b, ious, 0.5, False)
    assert rv
    assert rv == b[0]


def test_annot_match_2():
    a = [Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )]
    b = [
        Annotation(1, 1, [], 1, None, (0, 0, 50, 50), 1.0),
        Annotation(2, 1, [], 1, None, (5, 5, 50, 50), 1.0),
        Annotation(3, 1, [], 1, None, (10, 10, 50, 50), 1.0),
        Annotation(4, 1, [], 1, None, (0, 0, 45, 40), 1.0),
    ]
    ious = precalculate_combinatorial_ious(a, b, Annotation.IoUMethod.Box, show_progress=False)
    rv = match_pred_to_truth(a[0], b, ious, 0.5, False)
    assert rv
    assert rv == b[0]


def test_annot_match_3():
    a = [Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )]
    b = [
        Annotation(1, 1, [], 1, None, (6.25, 6.25, 56.25, 56.25), 1.0),
        Annotation(2, 1, [], 1, None, (25, 25, 50, 50), 1.0),
    ]
    ious = precalculate_combinatorial_ious(a, b, Annotation.IoUMethod.Box, show_progress=False)
    print(ious)
    rv = match_pred_to_truth(a[0], b, ious, 0.5, False)
    assert rv
    assert rv == b[0]


def test_annot_match_4():
    a = [Annotation(
        1, 1, [], 1, None, (0, 0, 50, 50), 1.0
    )]
    b = [
        Annotation(1, 1, [], 1, None, (12.5, 12.5, 62.5, 62.5), 1.0),
    ]
    ious = precalculate_combinatorial_ious(a, b, Annotation.IoUMethod.Box, show_progress=False)
    rv = match_pred_to_truth(a[0], b, ious, 0.5, False)
    assert not rv