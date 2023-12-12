
import torch


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
        
        
def save_checkpoint(model, optimizer, epoch, filename):
    # Create directory if it does not exist
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)


# def evaluate_model(model, test_features, test_output, criterion, device):
    # model.eval()  # Set the model to evaluation mode
    # predictions = []

    # with torch.no_grad():
    #     # Wrap the range function with tqdm for a progress bar
    #     progress_bar = tqdm(range(len(test_output)), desc='Evaluating', unit='batch')

    #     for i in progress_bar:
    #         text_input_ids = model.tokenize(test_features['text'][i])

    #         non_text_feature_inputs = []
    #         for feature_type in list(test_features.keys())[1:]:
    #             non_text_feature_inputs.append(torch.tensor(test_features[feature_type][i]).to(device))

    #         predicted = model(text_input_ids, non_text_feature_inputs, label_ids=None)

    #         predictions.extend(predicted)

    # accuracy = np.mean(np.array(predictions) == np.array(test_output))
    # print(f'Test Accuracy: {accuracy:.4f}')
    
    # # Save predictions and actuals to a file
    # with open('predictions_actuals.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Prediction', 'Actual'])
    #     for pred, act in zip(predictions, test_output):
    #         writer.writerow([pred, act])
            
    # return accuracy
# def train_model(model, train_features, train_output, test_features, test_output, optimizer, criterion, device, num_epochs, checkpoint_path):
    
    # os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # model.train()
    # best_loss = float('inf')
    
    # for epoch in range(num_epochs):
    #     total_loss = 0

    #     # Wrap the range function with tqdm for a progress bar
    #     progress_bar = tqdm(range(len(train_output)), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

    #     for i in progress_bar:
    #         text_input_ids = model.tokenize(train_features['text'][i])
    #         label_ids = model.tokenize(train_output[i])

    #         # Prepare non-text features
    #         non_text_feature_inputs = []
    #         if len(train_features.keys()) > 1:
    #             for feature_type in list(train_features.keys())[1:]:
    #                 non_text_feature_inputs.append(torch.tensor(train_features[feature_type][i]).to(device))
                    
    #         optimizer.zero_grad()
    #         loss = model(text_input_ids, non_text_feature_inputs, label_ids)
    #         total_loss += loss.item()

    #         loss.backward()
    #         optimizer.step()
    #         del text_input_ids, non_text_feature_inputs, label_ids
    #         torch.cuda.empty_cache()

    #         # Update progress bar
    #         progress_bar.set_postfix({'loss': total_loss / (i + 1)})

    #     average_loss = total_loss / len(train_output)
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}')

    #     # Save checkpoint if it's the best model so far
    #     if average_loss < best_loss:
    #         best_loss = average_loss
    #         checkpoint_filename = os.path.join(checkpoint_path, f'model_checkpoint_epoch_{epoch+1}.pth')
    #         save_checkpoint(model, optimizer, epoch, checkpoint_filename)
            
    #     accuracy = evaluate_model(model, test_features, test_output, criterion, device)

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
    