from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from info import *
import os

class MMIDataset(Dataset):
    # Multi-modal Individual Dataset
    def __init__(
        self, 
        feature_list, 
        data_type: str, 
        dataset_name: str, 
        dataset_rootdir: str = '../meta/',
        data_split=[0], 
        nrows: int = -1, 
        pred_mode: str = 'seq2seq',
        data_mean: pd.Series = None,
        data_std: pd.Series = None,
        label_mean: float = None,
        label_std: float = None,
        sample_frac: float = 1.0,
        slice_range: tuple = None
    ):
        
        eps = 1e-5

        self.dataset_name = dataset_name

        print(f"Loading Dataset {dataset_name},", end=' ')

        dataset_files = [dataset_rootdir + dataset_name + f'_{idx}_{data_type}.csv'
                   for idx in data_split]
                   
        df_list = [pd.read_csv(dataset_file)
                   for dataset_file in dataset_files] if nrows <= 0 else \
                  [pd.read_csv(dataset_file, nrows=nrows)
                   for dataset_file in dataset_files]
                  
        df = pd.concat(df_list, ignore_index=True)
        
        features = [ft for ft in df.columns if not (ft.startswith('meta') or ft == 'y' or ft == 'sentence')]
        data = df[features]
        data = data.dropna(axis='columns')
        self.data_mean = data.mean()
        self.data_std = data.std()

        data = (data - self.data_mean) / (self.data_std + eps)

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
        
        # Create label
        if pred_mode == 'seq2seq':
            if self.task_type == 'C':
                self.label = label
            else:
                self.label = [f"{float(number):.2f}" for number in label]
        else:
            if self.task_type == 'C':
                self.label = [int(y) for y in df['y']]
                int_array = np.array(self.label).reshape(-1, 1)

                encoder = OneHotEncoder(sparse=False)
                self.label = encoder.fit_transform(int_array)
            else:
                self.label = [float(y) for y in df['y']]
                self.label = torch.tensor(self.label).unsqueeze(1) 
                self.label_mean = self.label.mean()
                self.label_std = self.label.std()

                self.label = (self.label - self.label_mean) / (self.label_std + eps)

        self.dataset_size = len(self.label)
        print(f"size: {self.dataset_size}")
        
        self.all_modalities.append('text')
        task_instruction = DATASET_INSTRUCTION[dataset_name] 
        
        if 'sentence' in df.columns:
            self.data['text'] = [sent + "<SEP> Task: Given the input, " + task_instruction for sent in list(df['sentence'])]
        else:
            self.data['text'] = ["<SEP> Task: Given the input, " + task_instruction] * self.dataset_size
        
        print()
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        label = self.label[idx]

        feature = {}
        for mod in self.all_modalities:
            feature[mod] = self.data[mod][idx]

        return feature, label, self.dataset_name, self.task_type
    
    def return_stats(self):
            return self.data_mean, self.data_std, self.label_mean, self.label_std 

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


