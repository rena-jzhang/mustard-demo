from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

from info import *
import os

class MMIDataset(Dataset):
    # Multi-modal Individual Dataset
    def __init__(self, feature_list, data_type: str, dataset_name: str, dataset_rootdir: str = '../meta/',
                 data_split=None, nrows: int = -1, filter_dim_coordination=False, sample_frac: float = 1.0,
                 slice_range: tuple = None):
        
        eps = 1e-5
        # assert dataset_name in ALL_DATASETS
        # assert data_type in ALL_DATA_TYPES
        self.dataset_name = dataset_name
        print(f"Loading Dataset {dataset_name},", end=' ')

        # if dataset_name in ['vreed_av']:
        #     dataset_rootdir = dataset_rootdir.replace('meta', 'meta_vreed')
        # elif dataset_name in ['iemocap_arousal', 'iemocap_valence']:
        #     dataset_rootdir = dataset_rootdir.replace('meta', 'meta_iemocap')

        # Dynamically identify files that match the pattern
        file_pattern = f'_{data_type}.csv'
        all_files = os.listdir(dataset_rootdir)
        dataset_files = [file for file in all_files if file.startswith(dataset_name) and file.endswith(file_pattern)]

        # Read the identified files into DataFrame(s)
        if nrows <= 0:
            df_list = [pd.read_csv(os.path.join(dataset_rootdir, file)) for file in dataset_files]
        else:
            df_list = [pd.read_csv(os.path.join(dataset_rootdir, file), nrows=nrows) for file in dataset_files]

        df = pd.concat(df_list, ignore_index=True)
        
        # df = df[slice_range[0]:slice_range[1]] if slice_range is not None else df
        # df = df.sample(frac=sample_frac, random_state=1706)

        features = [ft for ft in df.columns if not (ft.startswith('meta') or ft == 'y')]
        data = df[features]
        data = data.dropna(axis='columns')
        data = (data - data.mean()) / (data.std() + eps)

        modalities = list(set([ft.split('_')[0] for ft in data.columns]))

        self.data = {}
        self.feature_size = {}
        self.all_modalities = []
        
        for modality in modalities:
            if modality not in feature_list:
                continue
            feature_names = [ft for ft in data.columns if ft.startswith(modality)]
            feature_df = data[feature_names]
            data_mod = torch.tensor(feature_df.values).float()
            self.data[modality] = data_mod
            self.feature_size[modality] = len(self.data[modality][0])
            self.all_modalities.append(modality)
        
        label = list(df['y'])
        
        self.task_type = DATASET_TASK[dataset_name]
                
        if self.task_type == 'C':
            self.label = label
        else:
            self.label = [f"{float(number):.2f}" for number in label]
            
        self.dataset_size = len(self.label)
        print(f"size: {self.dataset_size}")
        
        self.all_modalities.append('text')
        task_instruction = DATASET_INSTRUCTION[dataset_name]
        self.data['text'] = [task_instruction] * self.dataset_size
        
        print()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        label = self.label[idx]

        feature = {}
        for mod in self.all_modalities:
            feature[mod] = self.data[mod][idx]

        return feature, label, self.dataset_name, self.task_type

def get_datasets(args) -> Dict[str, MMIDataset]:
    assert not args.multitask

    return {data_type: MMIDataset(data_type, args.dataset_name, args.dataset_dir, args.dataset_split)
            for data_type in ALL_DATA_TYPES}


def get_multitask_datasets(args) -> Tuple[Dict[str, List[MMIDataset]], Dict[str, MMIDataset]]:
    assert args.multitask
    # assert all(dataset_name in ALL_DATASETS for dataset_name in args.dataset_name_list)

    print("\n[Loading validation datasets]")
    multitask_validation_datasets = {dataset_name: MMIDataset('validation', dataset_name, args.dataset_dir, args.dataset_split)
                                     for dataset_name in args.dataset_name_list}

    print("\n[Loading training datasets]")
    multitask_training_datasets = {dataset_name: [] for dataset_name in args.dataset_name_list}

    if args.balanced:
        for dataset_name in args.dataset_name_list:
            start_idx = 0
            end_idx = args.per_dataset_size
            for _ in range(args.max_num):
                try:
                    dataset = MMIDataset('training', dataset_name, args.dataset_dir, args.dataset_split,
                                         slice_range=(start_idx, end_idx))
                except:
                    break
                multitask_training_datasets[dataset_name].append(dataset)

                start_idx = end_idx
                end_idx += args.per_dataset_size

    else:
        NotImplementedError()

    return multitask_training_datasets, multitask_validation_datasets
