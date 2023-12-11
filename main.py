import torch
import torch.nn as nn
from torchvision.models import resnet50
from collections import defaultdict
from config import CONFIG_BY_KEY
from data_loader import DataPreper, DataHelper
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import gpu_monitor, save_checkpoint, prompt_eng
from tqdm import tqdm  # Import tqdm
import csv


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

                self.modules[feature_type]['embedding_transform'] = nn.Linear(283, self.model.config.hidden_size).double().to(device)

    def tokenize(self, text_input):
        return self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        
    def forward(self, text_input_ids, non_text_features, label_ids = None):
        # process text
        input_embeddings = self.model.get_input_embeddings()
        text_embeddings = input_embeddings(text_input_ids)

        # Process non-text features
        feature_inputs = []
        for i, feature_type in enumerate(self.feature_types):
            mode = self.feature_modes.get(feature_type)

            embedding_transform = self.modules[feature_type]['embedding_transform']
            
            if mode == 'raw':
                encoder = self.modules[feature_type]['encoder']

                with torch.no_grad():
                    feature_input = encoder(non_text_features[i])
                feature_input = torch.flatten(feature_input, start_dim=1)

                feature_embeddings = embedding_transform(feature_input)
                feature_inputs.append(feature_embeddings.unsqueeze(1))
            
            elif mode == 'precomputed':
                if non_text_features[i].dim() == 1:
                    feature_input = non_text_features[i].unsqueeze(0).unsqueeze(0)
                else:
                    feature_input = non_text_features[i].unsqueeze(1)
                                    
                feature_embeddings = embedding_transform(feature_input)
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

        # Checking for T5 model
        for name, param in self.model.named_parameters():
            print(f"self.model.{name}: {param.requires_grad}")

        # Checking for other modules like encoders and embedding_transform
        for feature_type in self.feature_types:
            for module_name, module in self.modules[feature_type].items():
                for name, param in module.named_parameters():
                    print(f"self.modules[{feature_type}][{module_name}].{name}: {param.requires_grad}")
def evaluate_model(model, test_features, test_output, criterion, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        # Wrap the range function with tqdm for a progress bar
        progress_bar = tqdm(range(len(test_output)), desc='Evaluating', unit='batch')

        for i in progress_bar:
            text_input_ids = model.tokenize(test_features['text'][i])

            non_text_feature_inputs = []
            for feature_type in list(test_features.keys())[1:]:
                non_text_feature_inputs.append(torch.tensor(test_features[feature_type][i]).to(device))

            predicted = model(text_input_ids, non_text_feature_inputs, label_ids=None)

            predictions.extend(predicted)

    accuracy = np.mean(np.array(predictions) == np.array(test_output))
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Save predictions and actuals to a file
    with open('predictions_actuals.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prediction', 'Actual'])
        for pred, act in zip(predictions, test_output):
            writer.writerow([pred, act])

    # # Confusion Matrix
    # print("Confusion Matrix:")
    # print(confusion_matrix(actuals, predictions))

    # # Classification Report
    # print("Classification Report:")
    # print(classification_report(actuals, predictions))

    return accuracy


def train_model(model, train_features, train_output, test_features, test_output, optimizer, criterion, device, num_epochs, checkpoint_path):
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0

        # Wrap the range function with tqdm for a progress bar
        progress_bar = tqdm(range(len(train_output)), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for i in progress_bar:
            text_input_ids = model.tokenize(train_features['text'][i])
            label_ids = model.tokenize(train_output[i])

            # Prepare non-text features
            non_text_feature_inputs = []
            if len(train_features.keys()) > 1:
                for feature_type in list(train_features.keys())[1:]:
                    non_text_feature_inputs.append(torch.tensor(train_features[feature_type][i]).to(device))
                    
            optimizer.zero_grad()
            loss = model(text_input_ids, non_text_feature_inputs, label_ids)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            del text_input_ids, non_text_feature_inputs, label_ids
            torch.cuda.empty_cache()

            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / (i + 1)})

        average_loss = total_loss / len(train_output)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}')

        # Save checkpoint if it's the best model so far
        if average_loss < best_loss:
            best_loss = average_loss
            checkpoint_filename = os.path.join(checkpoint_path, f'model_checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_filename)
            
        accuracy = evaluate_model(model, test_features, test_output, criterion, device)


def train_io(config, data, train_index, test_index):
    train_input, train_output = data.get_split(train_index)
    test_input, test_output = data.get_split(test_index)

    datahelper = DataHelper(train_input, train_output, test_input, test_output, config, data)

    train_features = {}
    test_features = {}

    if config.use_target_text:
        if config.use_bert:
            train_features['text'] = datahelper.get_target_bert_feature(mode="train")
            test_features['text'] = datahelper.get_target_bert_feature(mode="test")
        else:
            train_features['text'] = datahelper.vectorize_utterance(mode="train")
            test_features['text'] = datahelper.vectorize_utterance(mode="test")
            
    if config.use_target_video:
        train_features['video'] = datahelper.get_target_video_pool(mode="train")
        test_features['video'] = datahelper.get_target_video_pool(mode="test")
        
    if config.use_target_audio:
        train_features['audio'] = datahelper.get_target_audio_pool(mode="train")
        test_features['audio'] = datahelper.get_target_audio_pool(mode="test")

    # Check if any modality is being used
    if all(len(features) == 0 for features in train_features.values()):
        raise ValueError("Invalid modalities")

    return train_features, train_output, test_features, test_output
    
    
def proprocess_output(train_output, test_output, class_mapping):
    train_output = [class_mapping[i] for i in train_output]
    test_output = [class_mapping[i] for i in test_output]
    return train_output, test_output


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
        
        # features a dict of features self.data[mod] = torch.tensor(padded features)
        # self.dataset_name
        # self.task_type
        # text: self.data['language'] = a tensor: tokenized(prompt_enged(list of texts,dataset_name, task_type))
        # label: label = a tensor: tokenized(preprocessed(list of labels, dataset_name, task_type))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        label = self.label[idx]

        feature = {}
        for mod in self.all_modalities:
            feature[mod] = self.data[mod][idx]

        # return text, feature, label, self.dataset_name, self.task_type
        return feature, label, self.dataset_name, self.task_type

def train(config, data):
    all_indices = data.get_all_indices_shuffled()

    split_point = int(len(all_indices) * 0.8)  
    train_index = all_indices[:split_point]
    test_index = all_indices[split_point:]

    # prepare data
    train_features, train_output, test_features, test_output = train_io(config=config, data=data, train_index=train_index, test_index=test_index)
    pirnt(test_features, test_output)
    exit()
    
    sarcasm_mapping = {
        0: "Non-Sarcastic",
        1: "Sarcastic"
    }
    train_output, test_output = proprocess_output(train_output, test_output, class_mapping =  sarcasm_mapping)

    template = "Examine the input and categorize it as 'Sarcastic' or 'Non-Sarcastic' in the context of binary sarcasm detection: "
    train_features, test_features = prompt_eng(train_features, test_features, template)  # add the instructions and prompts
    non_text_feature_modes = {'video': 'precomputed', 'audio': 'precomputed'}

    # prepare model
    tokenizer = T5Tokenizer.from_pretrained(LM_VERSION)
    model = TextFeatureOPTModel(LM_VERSION, list(non_text_feature_modes.keys()), tokenizer, feature_modes=non_text_feature_modes).to(device)

    model.print_gradients()

    # model = torch.nn.DataParallel(model)

    print("Prepared model")
    gpu_monitor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    train_model(model, train_features, train_output, test_features, test_output, optimizer, criterion, device, num_epochs, checkpoint_path = 'checkpoints/')
    
if __name__ == "__main__":
    
    torch.cuda.empty_cache()

    config = CONFIG_BY_KEY["tav"]
    
    print("Before running")
    gpu_monitor()
    
    data = DataPreper(config)
    train(config, data)