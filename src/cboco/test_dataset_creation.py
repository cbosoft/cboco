import os

from cboco.dataset import Dataset, Category

def test_dataset_creation_truly_empty():
    dataset = Dataset.empty([])
    assert dataset

def test_dataset_creation_truly_with_categories():
    dataset = Dataset.empty([Category(1, 'foo'), Category(2, 'bar')])
    assert len(dataset.categories) == 2
    assert dataset.categories[0].id == 1
    assert dataset.categories[1].id == 2

def test_dataset_from_json():
    dataset_a = Dataset.from_json(os.path.join('test_data', 'A.json'))
    assert len(dataset_a.images) == 5
    dataset_b = Dataset.from_json(os.path.join('test_data', 'B.json'))
    assert len(dataset_b.images) == 5