from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

import os
import numpy as np
from sklearn.model_selection import train_test_split

import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models


def audio_file_processor(audio_folder):
    # Add your audio processing logic here
    return []

def text_file_processor(text_folder):
    # Add your text processing logic here
    return []

def label_file_processor(label_folder):
    # Add your label processing logic here
    return []

# lists of features
def process_dataset(dataset_folders):
    data = {'video': [], 'audio': [], 'text': [], 'label': []}
    for modality, folder in dataset_folders.items():
        if modality == 'video':
            data['video'].extend(video_processor(folder))
        elif modality == 'audio':
            data['audio'].extend(audio_file_processor(folder))
        elif modality == 'text':
            data['text'].extend(text_file_processor(folder))
        elif modality == 'label':
            data['label'].extend(label_file_processor(folder))
    return data

# Split indices for a dataset
def indice_split(data, test_ratio=0.2):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    test_size = int(len(indices) * test_ratio)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    return train_indices, test_indices

# List of datasets, each represented by a dictionary
datasets = {
    'task1': {'video': 'path/to/video_folder1', 'audio': 'path/to/audio_folder1', 'text': 'path/to/text_folder1', 'label': 'path/to/label_folder1'},
    'task2': {'video': 'path/to/video_folder2', 'audio': 'path/to/audio_folder2', 'label': 'path/to/label_folder2'}
    # Add more datasets if necessary
}

# # Process each dataset and create train/test data
# for dataset in datasets:
#     processed_data = process_dataset(dataset)
    
#     train_indices, test_indices = indice_split(processed_data['label'])

#     # Create data for each set of indices
#     def create_data(indices):
#         data_input = {modality: [processed_data[modality][i] for i in indices] for modality in processed_data if modality != 'label'}
#         data_output = [processed_data['label'][i] for i in indices]
#         return data_input, data_output

#     train_input, train_output = create_data(train_indices)
#     test_input, test_output = create_data(test_indices)

class MMDataset(Dataset):
    def __init__(self, data_input, data_output, non_text_feature_modes, dataset_name, task_type, tokenizer):
        self.dataset_name = dataset_name
        self.task_type = task_type
        
        self.data = {}

        # Preprocess and tokenize all text data at once
        prompt_eng_texts = [self._preprocess(item[0], self.dataset_name, self.task_type) for item in data_input]
        tokenized_texts = tokenizer(prompt_eng_texts, return_tensors="pt", padding=True, truncation=True).input_ids
        self.data['text'] = tokenized_texts

        for feature_type, feature_mode in non_text_feature_modes.items():
            if feature_type == 'video':
                # TODO 
                audio_features = [item[4] for item in data_input]
                
                # precomputed
                if feature_mode != 'raw':
                    audio_tensor_list = [torch.tensor(features).permute(1, 0).mean(dim=0).unsqueeze(0) for features in audio_features]
                
                self.data['audio'] = audio_tensor_list
            elif feature_type == 'audio':
                # TODO
                video_features_files = [item[5] for item in data_input]
                
                # precomputed
                if feature_mode != 'raw':
                    video_tensor_list = [torch.tensor(features).mean(dim=0).unsqueeze(0) for features in video_features_files]
                
                
                self.data['video'] = video_tensor_list
                
        # Process labels
        processed_labels = [self._process_output(label, self.dataset_name, self.task_type) for label in data_output]
        # print("OUTPUT TEXTS: ", processed_labels)
        self.labels = tokenizer(processed_labels, return_tensors="pt", padding=True, truncation=True).input_ids

    def _preprocess(self, text, dataset_name, task_type):        
        template = "Examine the input and categorize it as 'Sarcastic' or 'Non-sarcastic' in the context of binary sarcasm detection: "
        return f"{template} {text}"
        # return text
        
    def _process_output(self, label, dataset_name, task_type):
        sarcasm_mapping = {
            0: "Non-Sarcastic",
            1: "Sarcastic"
            # 0: "non-sarcastic",
            # 1: "sarcastic"
        }
        return sarcasm_mapping[label]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        feature = {}
        for mod in self.data.keys():
            feature[mod] = self.data[mod][idx]

        return feature, label, self.dataset_name, self.task_type

def custom_collate_fn(batch):
    batched_data = {}
    labels = []
    dataset_names = []
    task_types = []

    # process features
    modalities = batch[0][0].keys() 
    for modality in modalities:
        
        features = [item[0][modality] for item in batch]
        if modality in ['audio', 'video']:  # Add other modalities requiring padding here
            batched_data[modality] = pad_sequence(features, batch_first=True, padding_value=0)
        else:  # For modalities that don't need padding
            batched_data[modality] = torch.stack(features, dim=0)

    # Process labels, dataset names, and task types
    labels = [item[1] for item in batch]
    labels_tensor = torch.stack(labels, dim=0)

    dataset_names = [item[2] for item in batch]
    task_types = [item[3] for item in batch]
    
    return batched_data, labels_tensor, dataset_names, task_types

class MMIDataset(Dataset):
    # Multi-modal Individual Dataset
    def __init__(self, data_type: str, dataset_name: str, dataset_rootdir: str = '../meta/',
                 data_split=[0], nrows: int = -1, filter_dim_coordination=False, sample_frac: float = 1.0,
                 slice_range: tuple = None):
        eps = 1e-5
        # assert dataset_name in ALL_DATASETS
        # assert data_type in ALL_DATA_TYPES
        self.dataset_name = dataset_name
        print(f"Loading Dataset {dataset_name},", end=' ')

        if dataset_name in ['vreed_av']:
            dataset_rootdir = dataset_rootdir.replace('meta', 'meta_vreed')
        elif dataset_name in ['iemocap_arousal', 'iemocap_valence']:
            dataset_rootdir = dataset_rootdir.replace('meta', 'meta_iemocap')

        df_list = [pd.read_csv(dataset_rootdir + dataset_name + f'_{idx}_{data_type}.csv')
                   for idx in data_split] if nrows <= 0 else \
                  [pd.read_csv(dataset_rootdir + dataset_name + f'_{idx}_{data_type}.csv', nrows=nrows)
                   for idx in data_split]

        df = pd.concat(df_list, ignore_index=True)
        df = df[slice_range[0]:slice_range[1]] if slice_range is not None else df
        df = df.sample(frac=sample_frac, random_state=1706)

        features = [ft for ft in df.columns if not (ft.startswith('meta') or ft == 'y')]
        data = df[features]
        data = data.dropna(axis='columns')
        data = (data - data.mean()) / (data.std() + eps)

        modalities = list(set([ft.split('_')[0] for ft in data.columns]))

        self.data = {}
        self.feature_size = {}
        self.all_modalities = []
        for modality in modalities:
            feature_names = [ft for ft in data.columns if ft.startswith(modality)]
            feature_df = data[feature_names]
            data_mod = torch.tensor(feature_df.values).float()
            # if filter_dim_coordination:
            #     if len(data_mod[0]) != MODALITY_FEATURE_SIZE[modality]:
            #         continue
            self.data[modality] = data_mod
            self.feature_size[modality] = len(self.data[modality][0])
            self.all_modalities.append(modality)

        label = torch.tensor(df[['y']].values)
        # self.task_type = DATASET_TASK[dataset_name]
        self.task_type = 'C'
        if self.task_type == 'C':
            self.label = F.one_hot(label, num_classes=4).float().squeeze(dim=-2)
            self.label_std = None
        else:
            self.label = (label - label.mean()) / (label.std() + eps)
            self.label_std = label.std().item()
            print(f"label std: {self.label_std},", end=' ')
        self.dataset_size = len(self.label)
        print(f"size: {self.dataset_size}")
        if self.task_type == 'C':
            # self.class_num = DATASET_CLASS_NUMBER[dataset_name]
            self.class_num = 4
        else:
            self.class_num = None
        print(self.label)
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
    # assert args.dataset_name in ALL_DATASETS
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
