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

class MMDataset(Dataset):
    def __init__(self, data_input, data_output, non_text_feature_modes, dataset_name, task_type, tokenizer):
        self.dataset_name = dataset_name
        self.task_type = task_type
        
        self.data = {}

        # Process input text: prompt + input text
        prompt_eng_texts = [self._preprocess(item[0], self.dataset_name, self.task_type) for item in data_input]
        # tokenized_texts = tokenizer(prompt_eng_texts, return_tensors="pt", padding=True, truncation=True).input_ids
        self.data['text'] = prompt_eng_texts

        for feature_type, feature_mode in non_text_feature_modes.items():
            if feature_type == 'video':

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
                
        # Process label -  Answer text
        processed_labels = [self._process_output(label, self.dataset_name, self.task_type) for label in data_output]
        # self.labels = tokenizer(processed_labels, return_tensors="pt", padding=True, truncation=True).input_ids
        self.labels = processed_labels

    def _preprocess(self, text, dataset_name, task_type):        
        template = "Examine the given input and categorize it as 'Sarcastic' or 'Non-sarcastic' in the context of binary sarcasm detection: "
        return f"{template} {text}"
        
    def _process_output(self, label, dataset_name, task_type):
        sarcasm_mapping = {
            # 0: "Answer: Non-Sarcastic",
            # 1: "Answer: Sarcastic"
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

        # feature: dict of tensor
        # label: text
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
        if modality not in ['text']:  # Add other modalities requiring padding here
            batched_data[modality] = pad_sequence(features, batch_first=True, padding_value=0)
        else:  
            # For modalities that don't need padding
            batched_data[modality] = features

    # Process labels, dataset names, and task types
    labels = torch.tensor([item[1] for item in batch])
    dataset_names = [item[2] for item in batch]
    task_types = [item[3] for item in batch]
    
    return batched_data, labels, dataset_names, task_types
