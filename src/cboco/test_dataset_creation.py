from cboco.dataset import Dataset, Category

def test_dataset_creation_truly_empty():
    dataset = Dataset.empty([])
    assert dataset

def test_dataset_creation_truly_with_categories():
    dataset = Dataset.empty([Category(1, 'foo'), Category(2, 'bar')])
    assert len(dataset.categories) == 2
    assert dataset.categories[0].id == 1
    assert dataset.categories[1].id == 2