import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

from config import CONFIG_BY_KEY
from data_loader import DataPreper, DataHelper
from utils import *
from mmidataset import MMDataset, custom_collate_fn
from torch.nn.utils.rnn import pad_sequence


import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_name, task_type = 'mustard', 'C'

# LM_VERSION = 'google/flan-t5-xxl'
# LM_VERSION = 't5-small'
# LM_VERSION = 'gpt2'
LM_VERSION = '../web-act/llm_ft/Mistral-7B-Instruct-v0.1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 3e-2
BATCH_SIZE = 2
TEST_BATCH_SIZE = 1
num_epochs = 20
seeds = [
    42, 
    2023, 
    2024
    ]
num_runs = 1
OVERFIT = False

class TextFeatureOPTModel(nn.Module):
    def __init__(self, model_name, non_text_feature_types, tokenizer, feature_modes):
        super(TextFeatureOPTModel, self).__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
        # Freeze llm
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
        
    def forward(self, features, device, labels=None):
        # Set pad_token_id if it's not already set
        if self.model.config.pad_token_id is None and hasattr(self.model.config, 'eos_token_id'):
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Process non-text features
        feature_inputs = []
        for i, feature_type in enumerate(self.feature_types):
            non_text_feature = features[feature_type].to(device)
            mode = self.feature_modes.get(feature_type)
            if mode == 'raw':
                encoder = self.modules[feature_type]['encoder']

                with torch.no_grad():
                    feature_input = encoder(non_text_feature)
                feature_input = torch.flatten(feature_input, start_dim=1).float()

                feature_embeddings = self.modules[feature_type]['embedding_transform'](feature_input)
                feature_inputs.append(feature_embeddings.unsqueeze(1).to(device))

            elif mode == 'precomputed':
                feature_embeddings = self.modules[feature_type]['embedding_transform'](non_text_feature.float())
                feature_inputs.append(feature_embeddings.to(device))

        text_input = features['text']

        # Process multimodal embeddings
        non_text_embeddings = self.fusion(feature_inputs)
        non_text_len = non_text_embeddings.shape[1]

        eos_token = self.tokenizer.eos_token
        bos_token = self.tokenizer.bos_token
        if labels is not None:
            bos_token_id = self.tokenizer.bos_token_id  # Get the BOS token ID

            input_ids = []
            label_ids = []
            attention_masks = []

            for question, answer in zip(text_input, labels):
                
                # Tokenize and concatenate question and answer and add some multimodal features in the begininig
                question_tokens = self.tokenizer.encode(question + eos_token, add_special_tokens=True)
                answer_tokens = self.tokenizer.encode(answer + eos_token, add_special_tokens=True)
                concatenated_tokens = question_tokens + answer_tokens

                # Create labels for training (shift right and use -100 for question part)
                label = [-100] * (non_text_len + len(question_tokens)) + answer_tokens

                # Create attention mask
                attention_mask = [1] * (len(concatenated_tokens) + non_text_len)

                # Append to lists
                input_ids.append(torch.tensor(concatenated_tokens))
                label_ids.append(torch.tensor(label))
                attention_masks.append(torch.tensor(attention_mask))

            # Convert lists to tensors and pad if necessary
            input_ids = pad_sequence(input_ids, batch_first=True).to(device)
            text_embeddings = self.model.get_input_embeddings()(input_ids)
            combined_embeddings = self.fusion([text_embeddings[:, :1, :], non_text_embeddings, text_embeddings[:, 1:, :]]).to(device)

            label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100).to(device)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0).to(device)
            #print(label_ids.shape, attention_masks.shape)
            #print(label_ids[0], attention_masks[0])
            
            print(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))
            loss = self.model(
                inputs_embeds=combined_embeddings, 
                labels=label_ids, 
                attention_mask=attention_masks, 
                return_dict=True
            ).loss
            print(loss)
            return loss
        
        else:
            # Prepare input for inference
            bos_token_id = self.tokenizer.bos_token_id  # Get the BOS token ID
            input_ids = []
            attention_masks = []

            for question in text_input:
                # Tokenize and concatenate question and add BOS token
                context_tokens = self.tokenizer.encode(question + eos_token + bos_token, add_special_tokens=True)

                # Create attention mask
                attention_mask = [1] * (len(context_tokens) + non_text_len)

                # Append to lists
                input_ids.append(torch.tensor(context_tokens))
                attention_masks.append(torch.tensor(attention_mask))

            # Convert lists to tensors and pad if necessary
            input_ids = pad_sequence(input_ids, batch_first=True).to(device)
            text_embeddings = self.model.get_input_embeddings()(input_ids)
            combined_embeddings = self.fusion([text_embeddings[:, :1, :], non_text_embeddings, text_embeddings[:, 1:, :]]).to(device)

            attention_masks = pad_sequence(attention_masks, batch_first=True).to(device)

            with torch.no_grad():
                # Create attention mask for combined embeddings
                outputs = self.model.generate(inputs_embeds=combined_embeddings.to(device), max_length=15, attention_mask=attention_masks)
            decoded_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(decoded_texts)
            return decoded_texts

            
    def fusion(self, multimodal_embeddings):
        return torch.cat(multimodal_embeddings, dim=1)
    
    def print_gradients(self):
        print("Parameters Gradients:")
        print('self.modules[audio][embedding_transform].bias.grad: ', self.modules["audio"]["embedding_transform"].bias.grad[:10])
                    
    def print_param(self):
        print("Parameters:")
        print('self.modules[audio][embedding_transform].bias: ', self.modules["audio"]["embedding_transform"].bias[:10])


def train(data, run_name):
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
    #non_text_feature_modes = {'audio': 'precomputed'}
        
    if OVERFIT:
        train_dataset = MMDataset(train_input[:50], train_output[:50], non_text_feature_modes, dataset_name, task_type, tokenizer)
    else:
        train_dataset = MMDataset(train_input, train_output, non_text_feature_modes, dataset_name, task_type, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn = custom_collate_fn)
    
    if OVERFIT:
        test_dataset = MMDataset(train_input[:50], train_output[:50], non_text_feature_modes, dataset_name, task_type, tokenizer)
    else:
        test_dataset = MMDataset(test_input, test_output, non_text_feature_modes, dataset_name, task_type, tokenizer)
        
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn = custom_collate_fn)

    # Example usage of data_loader in a training loop
    for batch in train_loader:
        features, labels, dataset_names, task_types = batch
        print(len(features['text'][0]), len(labels[0]))
        if 'video' in features.keys():
            print(features['video'].shape)
        if 'audio' in features.keys():
            print(features['audio'].shape)
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
            run_name = f'{LM_VERSION.split("/")[-1]}_nopretrain_{LR}_{BATCH_SIZE}_{seed}_{i}'   
            if OVERFIT:
                run_name += '_overfit'
                
            wandb.init(name=run_name)
            wandb.config = {
                "learning_rate": LR,
                "epochs": num_epochs,
                "batch_size": BATCH_SIZE
            }

            train(data, run_name)
        
            wandb.finish()