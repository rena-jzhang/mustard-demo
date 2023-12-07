
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
