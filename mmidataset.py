from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


# from train.info import *
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
