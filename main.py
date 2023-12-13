import os
import csv
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer

from config import CONFIG_BY_KEY
from data_loader import DataPreper, DataHelper
from utils import *

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LM_VERSION = 'google/flan-t5-xxl'
LM_VERSION = 't5-small'
# LM_VERSION = 'meta-llama/Llama-2-7b-hf'
# LM_VERSION = 'llama/llama-2-7b-hf'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 3e-3
BATCH_SIZE = 16
TEST_BATCH_SIZE = 64
num_epochs = 20

seed = 42
torch.manual_seed(seed)

class MyCustomDataset(Dataset):
    def __init__(self, data_input, data_output, dataset_name, task_type, tokenizer):
        self.dataset_name = dataset_name
        self.task_type = task_type
        
        self.data = {'text': [], 'audio': [], 'video': []}

        # Preprocess and tokenize all text data at once
        prompt_eng_texts = [self._preprocess(item[0], self.dataset_name, self.task_type) for item in data_input]
        # print("INPUT TEXTS: ", prompt_eng_texts)
        tokenized_texts = tokenizer(prompt_eng_texts, return_tensors="pt", padding=True, truncation=True).input_ids

        audio_features = [item[4] for item in data_input]
        audio_tensor_list = [torch.tensor(features).permute(1, 0).mean(dim=0).unsqueeze(0) for features in audio_features]
        video_features_files = [item[5] for item in data_input]
        video_tensor_list = [torch.tensor(features).mean(dim=0).unsqueeze(0) for features in video_features_files]

        # Convert lists to tensors
        self.data = {
            'text': tokenized_texts, # a 2d tensor
            # 'audio': pad_sequence(audio_tensor_list, batch_first=True, padding_value=0), # list of 2d tensors
            'audio': audio_tensor_list,
            # 'video':  pad_sequence(video_tensor_list, batch_first=True, padding_value=0)# list of 2d tensors
            'video': video_tensor_list
        }
        
        # Process labels
        processed_labels = [self._process_output(label, self.dataset_name, self.task_type) for label in data_output]
        # print("OUTPUT TEXTS: ", processed_labels)
        self.labels = tokenizer(processed_labels, return_tensors="pt", padding=True, truncation=True).input_ids

    def _preprocess(self, text, dataset_name, task_type):        
        template = "Examine the input and categorize it as 'Sarcastic' or 'Non-Sarcastic' in the context of binary sarcasm detection: "
        return f"{template} {text}"
        
    
    def _process_output(self, label, dataset_name, task_type):
        
        sarcasm_mapping = {
        0: "Non-Sarcastic",
        1: "Sarcastic"
        }
        return sarcasm_mapping[label]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        feature = {}
        for mod in self.data.keys():
            feature[mod] = self.data[mod][idx]

        # return text, feature, label, self.dataset_name, self.task_type
        return feature, label, self.dataset_name, self.task_type

def custom_collate_fn(batch):
    # Initialize containers for batched data and labels
    batched_data = {}
    labels = []
    dataset_names = []
    task_types = []

    # Assuming all items in the batch have the same keys
    modalities = batch[0][0].keys()  # Keys from the first item's data

    for modality in modalities:
        # Extract features for each modality and pad if necessary
        features = [item[0][modality] for item in batch]

        if modality in ['audio', 'video']:  # Add other modalities requiring padding here
            batched_data[modality] = pad_sequence(features, batch_first=True, padding_value=0)
        else:  # For modalities that don't need padding
            batched_data[modality] = torch.stack(features, dim=0)

    # Process labels, dataset names, and task types
    labels = [item[1] for item in batch]
    dataset_names = [item[2] for item in batch]
    task_types = [item[3] for item in batch]

    labels_tensor = torch.stack(labels, dim=0)
    # print(batched_data.items())
    # print(labels_tensor)

    return batched_data, labels_tensor, dataset_names, task_types

class TextFeatureOPTModel(nn.Module):
    def __init__(self, model_name, feature_types, tokenizer, feature_modes):
        super(TextFeatureOPTModel, self).__init__()

        if "t5" in model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze the T5 model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.feature_types = feature_types
        self.feature_modes = feature_modes 
        self.tokenizer = tokenizer

        # Initialize modules for different feature types
        self.modules = defaultdict(nn.ModuleDict)
        for feature_type in feature_types:
            if feature_type == 'video':
                if feature_modes.get(feature_type) == 'raw':
                    video_encoder = resnet50(pretrained=True).to(device)
                    video_encoder.fc = nn.Identity()

                    # Freeze video encoder parameters
                    for param in video_encoder.parameters():
                        param.requires_grad = False

                    self.modules[feature_type]['encoder'] = video_encoder

                self.modules[feature_type]['embedding_transform'] = nn.Linear(2048, self.model.config.hidden_size).to(device)
        
            elif feature_type == 'audio':
                if feature_modes.get(feature_type) == 'raw':
                    audio_encoder = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2),
                    ).to(device)

                    # Freeze audio encoder parameters
                    for param in audio_encoder.parameters():
                        param.requires_grad = False

                    self.modules[feature_type]['encoder'] = audio_encoder

                self.modules[feature_type]['embedding_transform'] = nn.Linear(283, self.model.config.hidden_size).to(device)

    def tokenize(self, text_input):
        return self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        
    def forward(self, features, device, label_ids = None):
        # process text
        text_input_ids = features['text'].to(device)
        input_embeddings = self.model.get_input_embeddings()
        text_embeddings = input_embeddings(text_input_ids)

        # Process non-text features
        feature_inputs = []
        for i, feature_type in enumerate(self.feature_types):
            # print(feature_type)
            non_text_feature = features[feature_type].to(device)
            mode = self.feature_modes.get(feature_type)
            
            if mode == 'raw':
                encoder = self.modules[feature_type]['encoder']

                with torch.no_grad():
                    feature_input = encoder(non_text_feature)
                feature_input = torch.flatten(feature_input, start_dim=1).float()

                feature_embeddings = self.modules[feature_type]['embedding_transform'](feature_input)
                feature_inputs.append(feature_embeddings.unsqueeze(1))
            
            elif mode == 'precomputed':
                              
                feature_embeddings = self.modules[feature_type]['embedding_transform'](non_text_feature.float())
                feature_inputs.append(feature_embeddings)

        # process mutlimodal embeddings
        multimodal_embeddings = [text_embeddings] + feature_inputs
        combined_embeddings = self.fusion(multimodal_embeddings)
        
        if label_ids is not None:
            loss = self.model(inputs_embeds=combined_embeddings.float(), labels=label_ids, return_dict=True).loss
            return loss
        else:
            with torch.no_grad():
                outputs = self.model.generate(inputs_embeds=combined_embeddings.float(), max_length=50)
            decoded_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return decoded_texts
        
    def fusion(self, multimodal_embeddings):
        return torch.cat(multimodal_embeddings, dim=1)
    
    def print_gradients(self):
        print("Parameters Gradients:")
        print('self.modules[audio][embedding_transform].bias.grad: ', self.modules["audio"]["embedding_transform"].bias.grad[:10])
                    
    def print_param(self):
        print("Parameters:")
        print('self.modules[audio][embedding_transform].bias: ', self.modules["audio"]["embedding_transform"].bias[:10])



def train(config, data):
    
    dataset_name, task_type = 'mustard', 'C'
    
    # wandb setup
    project_name = dataset_name
    run_name = f'{LM_VERSION.split("/")[-1]}_nopretrain_{LR}_{BATCH_SIZE}'   
    entity_name = 'rena-jzhang'  
    wandb.init(project=project_name, entity=entity_name, name = run_name)
    wandb.config = {
        "learning_rate": LR,
        "epochs": num_epochs,
        "batch_size": BATCH_SIZE
    }
    
    if "t5" in LM_VERSION:
        tokenizer = T5Tokenizer.from_pretrained(LM_VERSION)
    else:
        tokenizer = AutoTokenizer.from_pretrained(LM_VERSION)
        tokenizer.pad_token = tokenizer.eos_token
        
    # Split
    all_indices = data.get_all_indices_shuffled()
    split_point = int(len(all_indices) * 0.8)  
    train_index = all_indices[:split_point]
    test_index = all_indices[split_point:]
    
    ''' prev implementation
    # prepare data
    train_features, train_output, test_features, test_output = train_io(config=config, data=data, train_index=train_index, test_index=test_index)
    # test_features = {'text': list of texts, len, 'video': 2d array (len, dim), 'audio': 2d array (len, dim) }
    # test_output: a list of labels
    
    sarcasm_mapping = {
        0: "Non-Sarcastic",
        1: "Sarcastic"
    }
    train_output, test_output = proprocess_output(train_output, test_output, class_mapping =  sarcasm_mapping)

    template = "Examine the input and categorize it as 'Sarcastic' or 'Non-Sarcastic' in the context of binary sarcasm detection: "
    train_features, test_features = prompt_eng(train_features, test_features, template)  # add the instructions and prompts
    
    '''
    
    non_text_feature_modes = {'video': 'precomputed', 'audio': 'precomputed'}
    
    # --> new implementation
    train_input, train_output = data.get_split(train_index)
    test_input, test_output = data.get_split(test_index)
        
    train_dataset = MyCustomDataset(train_input, train_output, dataset_name, task_type, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn = custom_collate_fn)

    test_dataset = MyCustomDataset(test_input, test_output, dataset_name, task_type, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn = custom_collate_fn)

    # Example usage of data_loader in a training loop
    for batch in test_loader:
        features, labels, dataset_names, task_types = batch
        print(features['text'].shape, features['video'].shape, features['audio'].shape, labels.shape)
        break
    
    # prepare model
    model = TextFeatureOPTModel(LM_VERSION, list(non_text_feature_modes.keys()), tokenizer, feature_modes=non_text_feature_modes).to(device)
    model.float() 

    print("Prepared model")
    gpu_monitor()
        
    optimizer = prepare_optimizer(model, LR)
    
    train_model(model, train_loader, test_loader, optimizer, device, num_epochs, checkpoint_path = 'checkpoints/', run_name = run_name)
    
    wandb.finish()
if __name__ == "__main__":
    
    torch.cuda.empty_cache()

    config = CONFIG_BY_KEY["tav"]
    
    print("Before running")
    gpu_monitor()
    
    data = DataPreper(config)
    train(config, data)
    