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

from dataset import MMIDataset
from info import *
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_rootdir = '/results/twoertwe/meta/'  # Path to your dataset directory


# LM_VERSION = 'google/flan-t5-xxl'
# LM_VERSION = 't5-small'
LM_VERSION = 'gpt2'
# LM_VERSION = '../llama/llama-2-7b-hf'
# LM_VERSION = '../web-act/llm_ft/Mistral-7B-Instruct-v0.1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
seeds = [
    42, 
    # 2023, 
    # 2024
    ]
num_runs = 1

FROZEN_LLM = False

# special settings
OVERFIT = False
TEXT_ONLY = False
NO_TEXT = False

# LR = 1e-4
# BATCH_SIZE = 2
# TEST_BATCH_SIZE = 4

if FROZEN_LLM:
    LR = 1e-3
else:
    LR = 1e-4
    
BATCH_SIZE = 2
TEST_BATCH_SIZE = 32

class MultiSenseModel(nn.Module):
    def __init__(self, model_name, non_text_feature_types, tokenizer, feature_modes, feature_dims):
        super(MultiSenseModel, self).__init__()

        self.feature_types = non_text_feature_types
        
        self.feature_modes = feature_modes 
        self.tokenizer = tokenizer

        # Initialize modules for different feature types
        self.model = self.get_llm(model_name, frozen = FROZEN_LLM) 
        self.modules = defaultdict(nn.ModuleDict)
        
        for feature_type in self.feature_types:
            self.modules[feature_type]['linear_projection'] = nn.Linear(feature_dims[feature_type], self.model.config.hidden_size).to(device)
        
    def get_llm(self, model_name, frozen=False):
        if 'llama' in model_name:
            base_model = LlamaForCausalLM.from_pretrained(model_name)
        else:  
            base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if frozen:
            model = base_model
            for param in model.parameters():
                param.requires_grad = False
        else:
            if 'llama' in model_name:

                from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
                lora_config = LoraConfig(
                    r=16,
                    target_modules=["q_proj", "v_proj"],
                    task_type=TaskType.CAUSAL_LM,
                    lora_alpha=32,
                    lora_dropout=0.05
                )
                model = get_peft_model(base_model, lora_config)
                
            else:
            
                model = base_model

        if model.config.pad_token_id is None and hasattr(model.config, 'eos_token_id'):
            model.config.pad_token_id = model.config.eos_token_id
        return model


    def get_audio_encoder(self, frozen=True):
        audio_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        ).to(device)

        # Freeze audio encoder parameters
        if frozen:
            for param in audio_encoder.parameters():
                param.requires_grad = False
        return audio_encoder


    def get_video_encoder(self, frozen=True):
        video_encoder = resnet50(pretrained=True).to(device)
        video_encoder.fc = nn.Identity()

        # Freeze video encoder parameters
        if frozen:
            for param in video_encoder.parameters():
                param.requires_grad = False
        return video_encoder


    def tokenize(self, text_input):
        return self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    def patch_data(self, text_input, labels=None):
        input_ids = []
        label_ids = []
        attention_masks = []

        for idx in range(len(text_input)):
            question = text_input[idx]
            answer = labels[idx] if labels else None

            question_tokens = self.tokenizer.encode(
                self.tokenizer.bos_token + question + self.tokenizer.eos_token, 
                add_special_tokens=False
            )
            if answer:
                answer_tokens = self.tokenizer.encode(
                    self.tokenizer.bos_token + answer + self.tokenizer.eos_token, 
                    add_special_tokens=False
                )
            else:
                answer_tokens = self.tokenizer.encode(
                    self.tokenizer.bos_token, 
                    add_special_tokens=False
                )
                
            
            concatenated_tokens = question_tokens + answer_tokens

            # Create labels for training (shift right and use -100 for question part)
            label = [-100] * len(question_tokens) + answer_tokens

            # Create attention mask
            attention_mask = [1] * (len(concatenated_tokens))

            # Append to lists
            input_ids.append(torch.tensor(concatenated_tokens))
            label_ids.append(torch.tensor(label))
            attention_masks.append(torch.tensor(attention_mask))

        if labels:
            return input_ids, label_ids, attention_masks
        else:
            return input_ids, attention_masks


    def padding(self, input_ids=None, attention_masks=None, label_ids=None):
        if input_ids:
            reversed_input_ids = [input_id.flip(dims=[0]) for input_id in input_ids]
            reversed_input_ids = pad_sequence(
                reversed_input_ids, 
                batch_first=True,
                padding_value=0
            ).to(device) 
            input_ids = reversed_input_ids.flip(dims=[1])

        if attention_masks:
            reversed_attention_masks = [attention_mask.flip(dims=[0]) for attention_mask in attention_masks]
            reversed_attention_masks = pad_sequence(
                reversed_attention_masks, 
                batch_first=True, 
                padding_value=0
            ).to(device)
            attention_masks = reversed_attention_masks.flip(dims=[1])

        if label_ids:
            # print('BEFORE', label_ids)
            reversed_label_ids = [label_id.flip(dims=[0]) for label_id in label_ids]
            reversed_label_ids = pad_sequence(
                reversed_label_ids, 
                batch_first=True, 
                padding_value=-100
            ).to(device)
            label_ids = reversed_label_ids.flip(dims=[1])
            
            # print('AFTER', label_ids)
        return input_ids, label_ids, attention_masks


    def forward(self, features, device, labels=None):
        
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

                feature_embeddings = self.modules[feature_type]['linear_projection'](feature_input)
                feature_inputs.append(feature_embeddings.unsqueeze(1).to(device))

            elif mode == 'precomputed':
                # feature_embeddings = non_text_feature.float()
                # if feature_type != 'language':
                feature_embeddings = self.modules[feature_type]['linear_projection'](non_text_feature.float())
                if len(feature_embeddings.shape) == 2:
                    feature_embeddings = feature_embeddings.unsqueeze(1)
                feature_inputs.append(feature_embeddings.to(device))

        non_text_embeddings = self.fusion(feature_inputs)

        # process text feature 
        text_input = features['text']
        if labels is not None:
            
            # # Tokenize each text and print the results
            # label_ids = []
            # for text in labels[:5]:
            #     tokens = self.tokenizer.tokenize(text)
            #     token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            #     label_ids.append(token_ids)
            #     print(f"Text: {text}")
            #     print(f"Tokens: {tokens}")
            #     print(f"Token IDs: {token_ids}")
            #     print()
                
            # decoded_texts = self.tokenizer.batch_decode(
            #                     label_ids, 
            #         skip_special_tokens=True
            # )
            # print('SKIP: ', decoded_texts)
                
            # decoded_texts = self.tokenizer.batch_decode(
            #                     label_ids, 
            #         skip_special_tokens=False
            # )
            # print('NON SKIP: ', decoded_texts)
            # exit()
            
            
            input_ids, label_ids, attention_masks = self.patch_data(
                text_input=text_input, 
                labels=labels
            )
            
            input_ids, label_ids, attention_masks = self.padding(
                input_ids=input_ids, 
                attention_masks=attention_masks, 
                label_ids=label_ids
            )

            # Convert lists to tensors and pad if necessary
            text_embeddings = self.model.get_input_embeddings()(input_ids)
            
            # Add projection also for text
            # text_embeddings = self.modules['text']['linear_projection'](text_embeddings)
            
            fused_embeddings = self.fusion([
                non_text_embeddings, 
                text_embeddings
            ]).to(device)
            
            non_text_len = non_text_embeddings.shape[1]
            constant_label_ids = torch.full((label_ids.shape[0], non_text_len), -100).to(device)
            constant_attention_masks = torch.full((attention_masks.shape[0], non_text_len), 1).to(device)
            label_ids = torch.cat([constant_label_ids, label_ids], dim=1)
            
            attention_masks = torch.cat([constant_attention_masks, attention_masks], dim=1)
            #fused_embeddings = text_embeddings.to(device)
            
            #print(self.tokenizer.decode(
            #    input_ids[0], 
            #    skip_special_tokens=False
            #))
                        
            # print('LABEL: ',label_ids[:5])
            # exit()
            # print('Decoded LABELS: ', decoded_texts)
            
            loss = self.model(
                inputs_embeds=fused_embeddings, 
                labels=label_ids, 
                attention_mask=attention_masks, 
                return_dict=True
            ).loss
            return loss
        
        else:
            # Prepare input for inference
            input_ids, attention_masks = self.patch_data(
                text_input=text_input, 
                labels=None
            )
            input_ids, _, attention_masks = self.padding(
                input_ids=input_ids, 
                attention_masks=attention_masks
            )

            text_embeddings = self.model.get_input_embeddings()(input_ids)
            
            # text_embeddings = self.modules['text']['linear_projection'](text_embeddings)

            fused_embeddings = self.fusion([
                non_text_embeddings, 
                text_embeddings
            ]).to(device)
            
            non_text_len = non_text_embeddings.shape[1]
            constant_attention_masks = torch.full((attention_masks.shape[0], non_text_len), 1).to(device)
            attention_masks = torch.cat([constant_attention_masks, attention_masks], dim=1)
            #fused_embeddings = text_embeddings.to(device)

            #print(self.tokenizer.decode(
            #    input_ids[0], 
            #    skip_special_tokens=False
            #))
            #print(self.tokenizer.decode(
            #    input_ids[-1], 
            #    skip_special_tokens=False
            #))
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=fused_embeddings, 
                    attention_mask=attention_masks,
                    max_length=15, 
                )
                
                # print("Predicted Token IDs:", outputs[:5])

                decoded_texts = self.tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )
            return decoded_texts

            
    def fusion(self, multimodal_embeddings):
        return torch.cat(multimodal_embeddings, dim=1)
    
    def print_gradients(self):
        print("Parameters Gradients:")
        print('self.modules[audio][linear_projection].bias.grad: ', self.modules["audio"]["linear_projection"].bias.grad[:10])
                    
    def print_param(self):
        print("Parameters:")
        print('self.modules[audio][linear_projection].bias: ', self.modules["audio"]["linear_projection"].bias[:10])


def train(data, run_name, dataset_name):
    if 'llama' in LM_VERSION:
        tokenizer = LlamaTokenizer.from_pretrained(LM_VERSION)
    else:
        tokenizer = AutoTokenizer.from_pretrained(LM_VERSION)
        
    # Split
    # all_indices = data.get_all_indices_shuffled()
    # split_point = int(len(all_indices) * 0.8)  
    # train_index = all_indices[:split_point]
    # test_index = all_indices[split_point:]
    # train_input, train_output = data.get_split(train_index)
    # test_input, test_output = data.get_split(test_index)
    
        
    non_text_features = DATASET_MODALITY[dataset_name]
    
    if TEXT_ONLY:
        non_text_features = ['language']
    else:
        if NO_TEXT:
            non_text_features.remove('language')
            
            
    print('NON TEXT FEATURES: ', non_text_features)

        
    non_text_feature_modes = dict([(feature_type, 'precomputed') for feature_type in non_text_features])
            
    # if OVERFIT:
    #     train_dataset = MMDataset(train_input[:50], train_output[:50], non_text_feature_modes, dataset_name, task_type, tokenizer)
    # else:
    #     train_dataset = MMDataset(train_input, train_output, non_text_feature_modes, dataset_name, task_type, tokenizer)
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn = custom_collate_fn)
    
    # if OVERFIT:
    #     test_dataset = MMDataset(train_input[:50], train_output[:50], non_text_feature_modes, dataset_name, task_type, tokenizer)
    # else:
    #     test_dataset = MMDataset(test_input, test_output, non_text_feature_modes, dataset_name, task_type, tokenizer)
        
    # test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn = custom_collate_fn)
    
    # Creating datasets
    if OVERFIT:
        train_dataset = MMIDataset(feature_list=non_text_features, data_type='training', dataset_name=dataset_name, dataset_rootdir=dataset_rootdir, nrows=10)
        val_dataset = train_dataset
        test_dataset = train_dataset
    else:
        train_dataset = MMIDataset(feature_list=non_text_features, data_type='training', dataset_name=dataset_name, dataset_rootdir=dataset_rootdir)
        val_dataset = MMIDataset(feature_list=non_text_features, data_type='validation', dataset_name=dataset_name, dataset_rootdir=dataset_rootdir)
        test_dataset = MMIDataset(feature_list=non_text_features, data_type='test', dataset_name=dataset_name, dataset_rootdir=dataset_rootdir)
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=custom_collate_fn)
    
    
    # Example usage of data_loader in a training loop
    for batch in train_loader:
        features, labels, dataset_names, task_types = batch
        print('FEATURES IN LOADER: ', features.keys())
        # print(len(features['text'][0]), len(labels[0]))
        # for name, feature in features.items():
        #     print(name, feature[0])
        break
    
    # prepare model
    MODALITY_FEATURE_SIZE = {
        'vision': 125, 'acoustic': 140, 'language': 457,
        'eda': 62, 'ecg': 54, 'mocap': 330, 
    }
    
    if 'umeme' in dataset_name:
        MODALITY_FEATURE_SIZE['acoustic'] = 52
    
    model = MultiSenseModel(LM_VERSION, non_text_features, tokenizer, feature_modes=non_text_feature_modes, feature_dims=MODALITY_FEATURE_SIZE).to(device)
    model.float() 

    print("Prepared model")
    gpu_monitor()
        
    optimizer = prepare_optimizer(model, LR)
    train_model(model, train_loader, val_loader, test_loader, optimizer, device, num_epochs, checkpoint_path = f'checkpoints/{dataset_name}/', run_name = run_name)
    
if __name__ == "__main__":
    
    torch.cuda.empty_cache()

    config = CONFIG_BY_KEY["tav"]
    
    print("Before running")
    gpu_monitor()
    
    data = DataPreper(config)
    for seed in seeds:
        torch.manual_seed(seed)
        for i in range(num_runs):

            # dataset_name = 'sewa_valence'
            # dataset_name = 'recola_valence'
            for dataset_name in ALL_DATASETS:
                
                # wandb setup
                project_name = dataset_name
                run_name = f'{LM_VERSION.split("/")[-1]}_nopretrain_{LR}_{BATCH_SIZE}_{seed}_{i}'   
                if OVERFIT:
                    run_name += '_overfit'
                
                if not FROZEN_LLM:
                    run_name += '_unfreeze'
                    
                if TEXT_ONLY:
                    run_name += '_text-only'
                else:
                    if NO_TEXT:
                        run_name += '_no-text'

                wandb.init(project=project_name, name=run_name)
                wandb.config = {
                    "learning_rate": LR,
                    "epochs": num_epochs,
                    "batch_size": BATCH_SIZE
                }

                train(data, run_name, dataset_name = dataset_name)
            
                wandb.finish()