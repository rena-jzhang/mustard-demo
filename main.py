import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
# from transformers import LlamaForCausalLM, LlamaTokenizer

from config import CONFIG_BY_KEY
from data_loader import DataPreper, DataHelper
from utils import *
from mmidataset import MMDataset, custom_collate_fn

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_name, task_type = 'mustard', 'C'

# LM_VERSION = 'google/flan-t5-xxl'
LM_VERSION = 't5-small'
# LM_VERSION = 'meta-llama/Llama-2-7b-hf'
# LM_VERSION = 'llama/llama-2-7b-hf'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 3e-3
BATCH_SIZE = 16
TEST_BATCH_SIZE = 64
num_epochs = 50
seeds = [
    42, 
    2023, 
    2024
    ]
num_runs = 1

class TextFeatureOPTModel(nn.Module):
    def __init__(self, model_name, non_text_feature_types, tokenizer, feature_modes):
        super(TextFeatureOPTModel, self).__init__()

        if "t5" in model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze the T5 model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.feature_types = non_text_feature_types
        self.feature_modes = feature_modes 
        self.tokenizer = tokenizer

        # Initialize modules for different feature types
        self.modules = defaultdict(nn.ModuleDict)
        for feature_type in self.feature_types:
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


def train(data):
    
    
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
    train_input, train_output = data.get_split(train_index)
    test_input, test_output = data.get_split(test_index)
    
    non_text_feature_modes = {'video': 'precomputed', 'audio': 'precomputed'}
            
    train_dataset = MMDataset(train_input, train_output, non_text_feature_modes, dataset_name, task_type, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn = custom_collate_fn)

    test_dataset = MMDataset(test_input, test_output, non_text_feature_modes, dataset_name, task_type, tokenizer)
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
    
if __name__ == "__main__":
    
    torch.cuda.empty_cache()

    config = CONFIG_BY_KEY["tav"]
    
    print("Before running")
    gpu_monitor()
    
    data = DataPreper(config)
    for seed in seeds:
        torch.manual_seed(seed)
        for i in range(num_runs):
        
            # wandb setup
            project_name = dataset_name
            run_name = f'{LM_VERSION.split("/")[-1]}_nopretrain_{LR}_{BATCH_SIZE}_{seed}_{i}_50ep'   
            entity_name = 'rena-jzhang'  
            wandb.init(project=project_name, entity=entity_name, name = run_name)
            wandb.config = {
                "learning_rate": LR,
                "epochs": num_epochs,
                "batch_size": BATCH_SIZE
            }

            train(data)
        
            wandb.finish()
    