from info import *
from dataset import MMIDataset


def test_dataset_train_val_test_overlap():
    dataset_name = 'umeme_arousal'
    dataset_rootdir = '/results/twoertwe/meta/'  # Path to your dataset directory
    non_text_features = DATASET_MODALITY[dataset_name]
    train_dataset = MMIDataset(
        feature_list=non_text_features,
        data_type='training',
        dataset_name=dataset_name,
        dataset_rootdir=dataset_rootdir,
    )
    val_dataset = MMIDataset(
        feature_list=non_text_features,
        data_type='validation',
        dataset_name=dataset_name,
        dataset_rootdir=dataset_rootdir,
    )
    test_dataset = MMIDataset(
        feature_list=non_text_features,
        data_type='test',
        dataset_name=dataset_name,
        dataset_rootdir=dataset_rootdir,
    )

    train_text = []
    for data in train_dataset:
        train_text.append(data[0]['language'])

    assert len(train_dataset) != len(val_dataset)
    assert len(train_dataset) != len(test_dataset)
    
    for data in val_dataset:
        val_text = data[0]['language']
        assert val_text not in train_text
    
    for data in test_dataset:
        test_text = data[0]['language']
        
        if test_text in train_text:
            print(test_text)
    