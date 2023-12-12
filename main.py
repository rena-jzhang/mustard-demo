import os
import csv
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence

from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np

from config import CONFIG_BY_KEY
from data_loader import DataPreper, DataHelper
from utils import gpu_monitor, save_checkpoint, prompt_eng


# LM_VERSION = 'google/flan-t5-xxl'
LM_VERSION = 't5-small'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-4
seed = 42
torch.manual_seed(seed)

class TextFeatureOPTModel(nn.Module):
    def __init__(self, model_name, feature_types, tokenizer, feature_modes):
        super(TextFeatureOPTModel, self).__init__()
        # self.opt_model = AutoModelForCausalLM.from_pretrained(opt_model_name).to(device)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Freeze the T5 model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.feature_types = feature_types
        self.feature_modes = feature_modes 
        self.tokenizer = tokenizer

        self.modules = defaultdict(nn.ModuleDict)

        # Initialize modules for different feature types
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
        text_input_ids = features['text']
        input_embeddings = self.model.get_input_embeddings()
        text_embeddings = input_embeddings(text_input_ids)

        # Process non-text features
        feature_inputs = []
        for i, feature_type in enumerate(self.feature_types):
            # print(feature_type, )
            non_text_feature = features[feature_type].to(device)
            mode = self.feature_modes.get(feature_type)

            embedding_transform = self.modules[feature_type]['embedding_transform']
            
            if mode == 'raw':
                encoder = self.modules[feature_type]['encoder']

                with torch.no_grad():
                    feature_input = encoder(non_text_feature)
                feature_input = torch.flatten(feature_input, start_dim=1).float()

                feature_embeddings = embedding_transform(feature_input)
                feature_inputs.append(feature_embeddings.unsqueeze(1))
            
            elif mode == 'precomputed':
                # print('FEATURE DIM: ', non_text_feature.dim())
                # if non_text_feature.dim() == 1:
                #     feature_input = non_text_feature.unsqueeze(0).unsqueeze(0)
                # else:
                #     feature_input = non_text_feature.unsqueeze(1)
                                
                # feature_input = non_text_feature.mean(dim=1)
                feature_embeddings = embedding_transform(non_text_feature.float())
                feature_inputs.append(feature_embeddings)

        # Concatenate feature embeddings with text embeddings
        multimodal_embeddings = [text_embeddings] + feature_inputs
        
        combined_embeddings = self.fusion(multimodal_embeddings)
        
        # Handling both training and evaluation
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
        print("Parameters that require gradients:")

        # # Checking for T5 model
        # for name, param in self.model.named_parameters():
        #     print(f"self.model.{name}: {param.requires_grad}")

        # Checking for other modules like encoders and embedding_transform
        for feature_type in self.feature_types:
            for module_name, module in self.modules[feature_type].items():
                for name, param in module.named_parameters():
                    print(f"self.modules[{feature_type}][{module_name}].{name}: {param.requires_grad}")

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating', unit='batch')

        for batch in progress_bar:
            features, labels, _, _ = batch

            # Assuming 'text' is one of the modalities and labels are already tensorized
            # text_input_ids = features['text'].to(device)
            label_ids = labels.to(device)

            # Prepare non-text features
            # non_text_feature_inputs = dict([features[feature_type].to(device) for feature_type in features if feature_type != 'text']

            # Predict
            predicted = model(features, device, label_ids=None)

            # Store predictions and actual labels
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(label_ids.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(actuals))
    print(f'Test Accuracy: {accuracy:.4f}')

    # Save predictions and actuals to a file
    with open('predictions_actuals.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prediction', 'Actual'])
        for pred, act in zip(predictions, actuals):
            writer.writerow([pred, act])

    return accuracy

def train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for batch in progress_bar:
            # Unpack the batch
            features, labels, dataset_names, task_types = batch

            # Assuming 'text' is one of the modalities and labels are already tokenized
            # text_input_ids = features['text'].to(device)
            label_ids = labels.to(device)

            # Prepare non-text features (assuming they are already tensorized and moved to device in collate_fn)
            # non_text_feature_inputs = [features[feature_type].to(device) for feature_type in features if feature_type != 'text']

            optimizer.zero_grad()
            loss = model(features, device, label_ids)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / len(progress_bar)})

        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}')

        # Save checkpoint if it's the best model so far
        if average_loss < best_loss:
            best_loss = average_loss
            checkpoint_filename = os.path.join(checkpoint_path, f'model_checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_filename)
            
        # Evaluate model (ensure evaluate_model is adapted for DataLoader usage)
        evaluate_model(model,test_loader, device)
    
def proprocess_output(train_output, test_output, class_mapping):
    train_output = [class_mapping[i] for i in train_output]
    test_output = [class_mapping[i] for i in test_output]
    return train_output, test_output


class MyCustomDataset(Dataset):
    def __init__(self, data_input, data_output, dataset_name, task_type, tokenizer):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.data = {'text': [], 'audio': [], 'video': []}
        self.labels = []

        # Preprocess and tokenize all text data at once
        prompt_eng_texts = [self._preprocess(item[0], self.dataset_name, self.task_type) for item in data_input]
        tokenized_texts = tokenizer(prompt_eng_texts, return_tensors="pt", padding=True, truncation=True).input_ids

        # Assuming audio_features and video_features_file are in a format ready to be converted to tensors
        audio_features = [item[4] for item in data_input]
        audio_tensor_list = [torch.tensor(features).permute(1, 0) for features in audio_features]
        video_features_files = [item[5] for item in data_input]
        video_tensor_list = [torch.tensor(features) for features in video_features_files]

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
        self.labels = tokenizer(processed_labels, return_tensors="pt", padding=True, truncation=True).input_ids

    def _preprocess(self, text, dataset_name, task_type):
        # Define your preprocessing steps here
        # For example: concatenate the text with dataset_name and task_type
        return f"{text} - {dataset_name} - {task_type}"
    
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

    return batched_data, labels_tensor, dataset_names, task_types

def train(config, data):
    
    dataset_name, task_type = 'mustard', 'C'
        
    tokenizer = T5Tokenizer.from_pretrained(LM_VERSION)

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn = custom_collate_fn)

    test_dataset = MyCustomDataset(test_input, test_output, dataset_name, task_type, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn = custom_collate_fn)

    # Example usage of data_loader in a training loop
    for batch in test_loader:
        features, labels, dataset_names, task_types = batch
        print(features['text'].shape, features['video'].shape, features['audio'].shape, labels.shape)
        break
    
    # prepare model
    model = TextFeatureOPTModel(LM_VERSION, list(non_text_feature_modes.keys()), tokenizer, feature_modes=non_text_feature_modes).to(device)
    model.float() 
    model.print_gradients()

    # model = torch.nn.DataParallel(model)

    print("Prepared model")
    gpu_monitor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    

    # train_model(model, train_features, train_output, test_features, test_output, optimizer, criterion, device, num_epochs, checkpoint_path = 'checkpoints/')
    train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, checkpoint_path = 'checkpoints/')

if __name__ == "__main__":
    
    torch.cuda.empty_cache()

    config = CONFIG_BY_KEY["tav"]
    
    print("Before running")
    gpu_monitor()
    
    data = DataPreper(config)
    train(config, data)