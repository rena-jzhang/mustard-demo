import csv
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tqdm import tqdm
import wandb
import torch.nn as nn

def correlation(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)
    if torch.std(y_true) == 0 or torch.std(y_hat) == 0:
        return torch.tensor(0.0)
    correlation_matrix = torch.corrcoef(torch.stack([y_true, y_hat]))
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def correlation_loss(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    corr = correlation(y_true, y_hat)
    return torch.tensor(1.0) - corr

def concordance_correlation_coefficient(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    mean_y = torch.mean(y_true, dim=0)
    mean_y_hat = torch.mean(y_hat, dim=0)
    y_mean = y_true - mean_y
    y_hat_mean = y_hat - mean_y_hat
    cov = torch.mean(y_mean * y_hat_mean, dim=0)
    var = torch.var(y_true, dim=0, unbiased=False) + torch.var(y_hat, dim=0, unbiased=False)
    mse = (mean_y - mean_y_hat) ** 2

    # Check for division by zero
    denominator = var + mse
    if torch.all(denominator == 0):
        return torch.tensor(0.0)
    
    ccc = (2 * cov) / denominator
    return torch.mean(ccc)

def ccc_loss(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    ccc = concordance_correlation_coefficient(y_true, y_hat)
    return torch.tensor(1.0) - ccc

def cross_entropy_loss(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    criterion = nn.BCEWithLogitsLoss()
    return criterion(y_hat, y_true)

def loss_and_metric_for_dataset(dataset_name: str):
    if dataset_name in ['mosi_sentiment', 'mosei_happiness', 'mosei_sentiment']:
        return 'mae+corr', ['mae', 'corr']
    elif dataset_name in ['vreed_av']:
        return 'ce', ['acc', 'wf1']
    elif dataset_name in ['sewa_arousal', 'sewa_valence', 'umeme_arousal']:
        return 'mae+ccc', ['mae', 'ccc']
    elif dataset_name in ['recola_arousal', 'recola_valence']:
        return 'mae+ccc', ['mae', 'ccc']
    elif dataset_name in ['iemocap_valence', 'iemocap_arousal']:
        return 'mae+ccc', ['mae', 'ccc']
    else:
        raise ValueError(f"Wrong dataset name: {dataset_name}")


def get_loss_function(loss_fn_name: str, alpha: float = 0.9):
 
    if loss_fn_name == 'ce':
        return cross_entropy_loss
    elif loss_fn_name == 'mae':
        return nn.L1Loss()
    elif loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'mae+corr':
        return lambda x, y: alpha * nn.L1Loss()(x, y) + (1 - alpha) * correlation_loss(x, y)
    elif loss_fn_name == 'mae+ccc':
        return lambda x, y: alpha * nn.L1Loss()(x, y) + (1 - alpha) * ccc_loss(x, y)

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
    
    # Check for Pearson correlation and CCC
    if len(true_values) < 2 or len(predicted_values) < 2 or np.std(true_values) == 0 or np.std(predicted_values) == 0:
        # Handle edge case for insufficient data or zero variance
        pearson_corr = 0
        ccc = 0
    else:
        pearson_corr, _ = pearsonr(true_values, predicted_values)
        denominator = (var_true + var_predicted + (mean_true - mean_predicted) ** 2)
        if denominator == 0:
            ccc = 0
        else:
            ccc = (2 * pearson_corr * np.sqrt(var_true) * np.sqrt(var_predicted)) / denominator

    # Check for RMSE
    if len(true_values) == 0 or len(predicted_values) == 0 or len(true_values) != len(predicted_values):
        rmse = 0
    else:
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))

    pcc = pearson_corr

    return ccc, rmse, pcc

def evaluate_model(model, test_loader, device, epoch = None, run_name = None):

    return evaluate_model_pred_head(model, test_loader, device, task_type='regression', epoch=epoch, run_name=run_name)
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

def evaluate_model_pred_head(model, test_loader, device, task_type='regression', epoch=None, run_name=None):
    
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
            if task_type == 'regression':
                predictions.extend(predicted.flatten().tolist())            
                actuals.extend(labels.tolist())    
            else:
                predictions.extend(predicted)            
                actuals.extend(labels)   

    # Calculate accuracy, precision, recall, and F1 score
    
    # print(predictions[:3])
    # print(actuals[:3])

    # TODO
    if task_type == 'classification':

        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, average='macro')
        recall = recall_score(actuals, predictions, average='macro')
        f1 = f1_score(actuals, predictions, average='macro')
        
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        wandb.log({
        "eval_accuracy": accuracy, 
        "eval_precision": precision, 
        "eval_recall": recall, 
        "eval_f1": f1
    })

    elif task_type == 'regression':

        ccc, rmse, pcc = calculate_metrics(actuals, predictions)

        print(f'CCC: {ccc:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'PCC: {pcc:.4f}')
    
        wandb.log({
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
            writer.writerow([f'{pred:.4f}', f'{act:.4f}'])
    
def train_model(model, train_loaders, val_loader, test_loader, optimizer, device, num_epochs, checkpoint_path, run_name):
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_path, run_name + '.pth')

    for epoch in range(num_epochs):

        # Training phase
        model.train()
        total_train_loss = 0

        total_batches = 0

        zipped_loaders = zip(*[iter(loader) for loader in train_loaders])
        min_len = min(len(loader) for loader in train_loaders)
        train_progress_bar = tqdm(zipped_loaders, total=min_len, desc=f'Train Epoch {epoch+1}/{num_epochs}', unit='batch')

        for batches in train_progress_bar:
            optimizer.zero_grad()
            epoch_loss = 0
            for batch in batches:
                # print(batch)
                # print('done batch')
                features, labels, dataset_names, task_types = batch 
                loss = model(features, device, labels)  # You may need to adjust this call
                epoch_loss += loss
            
            epoch_loss.backward()  # Accumulate gradients for each task's batch before optimizer step
            optimizer.step()
            total_train_loss += epoch_loss.item()
            total_batches += 1

            torch.cuda.empty_cache()
            train_progress_bar.set_postfix({'loss': total_train_loss / total_batches})

        average_train_loss = total_train_loss / total_batches

        # train_progress_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}', unit='batch')

        # for batch in train_progress_bar:
        #     features, labels, dataset_names, task_types = batch
        #     optimizer.zero_grad()
        #     loss = model(features, device, labels)
        #     total_train_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
            
        #     del features, labels
        #     torch.cuda.empty_cache()
        #     train_progress_bar.set_postfix({'loss': total_train_loss / len(train_progress_bar)})

        # average_train_loss = total_train_loss / len(train_loader)
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

def prepare_optimizer(model, lr, pred_mode):
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

    if pred_mode == 'pred_head':
        linear_projection = model.output_layer
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
    