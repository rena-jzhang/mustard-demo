import csv
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tqdm import tqdm
import wandb


def calculate_metrics(true_values, predicted_values):
    """
    Calculate CCC, RMSE, and PCC.
    :param true_values: Array of true values
    :param predicted_values: Array of predicted values
    :return: Concordance Correlation Coefficient, Root Mean Squared Error, Pearson Correlation Coefficient
    """
    # Convert non-numeric values to NaN
    true_values = pd.to_numeric(true_values, errors='coerce')
    predicted_values = pd.to_numeric(predicted_values, errors='coerce')
    
    # Remove or impute NaNs (or use np.nanmean, np.nanvar, etc., to handle NaNs)
    valid_indices = ~np.isnan(true_values) & ~np.isnan(predicted_values)
    true_values = true_values[valid_indices]
    predicted_values = predicted_values[valid_indices]

    # Calculate CCC
    mean_true = np.mean(true_values)
    mean_predicted = np.mean(predicted_values)
    var_true = np.var(true_values)
    var_predicted = np.var(predicted_values)
    pearson_corr, _ = pearsonr(true_values, predicted_values)
    ccc = (2 * pearson_corr * np.sqrt(var_true) * np.sqrt(var_predicted)) / \
          (var_true + var_predicted + (mean_true - mean_predicted) ** 2)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))

    # PCC is the Pearson Correlation Coefficient
    pcc = pearson_corr

    return ccc, rmse, pcc

def evaluate_model(model, test_loader, device, epoch = None, run_name = None):
    model.eval() 
    predictions = []
    actuals = []
    result_folder_name = None
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating', unit='batch')

        for batch in progress_bar:
            features, labels, dataset_name, _ = batch
            result_folder_name = dataset_name[0]
            predicted = model(features, device)
            predictions.extend(predicted)            
            actuals.extend(labels)
            
    # Calculate accuracy, precision, recall, and F1 score
    actuals = [item.lower() for item in actuals]
    predictions = [res.lower() for res in predictions]
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='macro')
    recall = recall_score(actuals, predictions, average='macro')
    f1 = f1_score(actuals, predictions, average='macro')
    ccc, rmse, pcc = calculate_metrics(actuals, predictions)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'CCC: {ccc:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'PCC: {pcc:.4f}')
    
    wandb.log({
        "eval_accuracy": accuracy, 
        "eval_precision": precision, 
        "eval_recall": recall, 
        "eval_f1": f1,
        "eval_ccc": ccc,
        "eval_rmse": rmse,
        "eval_pcc": pcc
    })
    
    # Save predictions and actuals to a file
    folder_name = f"results/{result_folder_name}/{run_name}/"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if epoch is not None:
        file_name = os.path.join(folder_name, f'predictions_actuals_{epoch}.csv')
    else:
        file_name = os.path.join(folder_name, 'predictions_actuals.csv')

    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Prediction', 'Actual'])
        for pred, act in zip(predictions, actuals):
            writer.writerow([pred, act])

def save_checkpoint(model, optimizer, epoch, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)
    
def train_model(model, train_loader, val_loader, test_loader, optimizer, device, num_epochs, checkpoint_path, run_name):
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_path, 'best_model.pth')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}', unit='batch')

        for batch in train_progress_bar:
            features, labels, dataset_names, task_types = batch
            optimizer.zero_grad()
            loss = model(features, device, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            del features, labels
            torch.cuda.empty_cache()
            train_progress_bar.set_postfix({'loss': total_train_loss / len(train_progress_bar)})

        average_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_train_loss:.4f}')
        wandb.log({"epoch": epoch, "train_loss": average_train_loss, "lr": optimizer.param_groups[0]['lr']})

        # Validation phase with progress bar
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{num_epochs}', unit='batch')

        with torch.no_grad():
            for batch in val_progress_bar:
                features, labels, dataset_names, task_types = batch
                val_loss = model(features, device, labels)
                total_val_loss += val_loss.item()

                del features, labels
                torch.cuda.empty_cache()

                val_progress_bar.set_postfix({'val_loss': total_val_loss / len(val_progress_bar)})

        average_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}')
        wandb.log({"val_loss": average_val_loss})

        # # Save the best model based on validation loss
        # if average_val_loss < best_val_loss:
        #     best_val_loss = average_val_loss
        #     torch.save(model.state_dict(), best_model_path)
        
        evaluate_model(model, test_loader, device, epoch, run_name)  # -1 indicates test evaluation

    # # Load the best model and evaluate on the test set
    # model.load_state_dict(torch.load(best_model_path))
    # evaluate_model(model, test_loader, device, None, run_name)  # -1 indicates test evaluation

def proprocess_output(train_output, test_output, class_mapping):
    train_output = [class_mapping[i] for i in train_output]
    test_output = [class_mapping[i] for i in test_output]
    return train_output, test_output

def prepare_optimizer(model, lr):
    trainable_params = []
    for param in model.model.parameters():
        if param.requires_grad:
            trainable_params.append(param)
    for feature_type in model.modules.keys():
        linear_projection = model.modules[feature_type]['linear_projection']
        if linear_projection is not None:
            for param in linear_projection.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    return optimizer

def postpros(res):
    if res in ['sarcasm', 'sarcastic', '(sarcastic)']:
        return 'sarcastic'
    return res

def prompt_eng(train_features, test_features, template):
    train_features['text'] = [f"{template} {text}" for text in train_features['text']]
    test_features['text'] = [f"{template} {text}" for text in test_features['text']]

    return train_features, test_features

def gpu_monitor():
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the ID of the current GPU
        device_id = torch.cuda.current_device()

        # Get the name of the current GPU
        gpu_name = torch.cuda.get_device_name(device_id)

        # Get the total memory of the current GPU
        total_memory = torch.cuda.get_device_properties(device_id).total_memory

        # Convert bytes to megabytes
        total_memory_in_MB = total_memory / (1024**2)

        # Get the current memory usage
        current_memory_allocated = torch.cuda.memory_allocated(device_id)
        current_memory_allocated_in_MB = current_memory_allocated / (1024**2)

        # Get the current memory cached
        current_memory_cached = torch.cuda.memory_reserved(device_id)
        current_memory_cached_in_MB = current_memory_cached / (1024**2)

        # Calculate free memory
        free_memory_in_MB = total_memory_in_MB - current_memory_allocated_in_MB

        print(f"GPU: {gpu_name}")
        print(f"Total GPU Memory: {total_memory_in_MB:.2f} MB")
        print(f"Currently Allocated Memory: {current_memory_allocated_in_MB:.2f} MB")
        print(f"Currently Cached Memory: {current_memory_cached_in_MB:.2f} MB")
        print(f"Free Memory: {free_memory_in_MB:.2f} MB")
    else:
        print("CUDA is not available. No GPU detected.")
        
        


def train_io(config, data, train_index, test_index):
    
    # train_input: a list of tuples (tex)
    # self.data_input.append((dataset_dict[id_]["utterance"], dataset_dict[id_]["speaker"],
    #                         dataset_dict[id_]["context"], dataset_dict[id_]["context_speakers"],
    #                         audio_features[id_] if audio_features else None,(dim, seqlen)
    #                         video_features_file[id_][()] if video_features_file else None,(seqlen, dim)
    #                         
    #                         context_video_features_file[id_][()] if context_video_features_file else None,
    #                         text_embeddings[idx] if text_embeddings else None,
    #                         context_embeddings[idx] if context_embeddings else None,
    #                         
    #                         dataset_dict[id_]["show"]))
    # train_output: a list of labels
    train_input, train_output = data.get_split(train_index)
    test_input, test_output = data.get_split(test_index)
    
    
    # print('TEST INPUT/n', test_input[0])

    # for j in range(1):
    #     for i in test_input[j]:
    #         # Check if the element is an instance of np.ndarray
    #         if isinstance(i, np.ndarray):
    #             print(i.shape)
    #         # elif isinstance(i, list):
    #         #     print(len(i))
    #         else:
    #             print(i)

    # print('TEST OUTPUT/n',test_output[0])
        
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
    