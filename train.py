import os
import csv
import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import datetime
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define transformations
transform_full = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to create a stratified split
def stratified_split(dataset, train_per_class=25000, val_per_class=5000):
    # Get targets (labels) from the dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # For ImageFolder dataset, targets are in dataset.imgs
        targets = [label for _, label in dataset.imgs]
    
    # Find indices for each class
    class_indices = {}
    for idx, label in enumerate(targets):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Shuffle indices for each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])
    
    # Create train and validation indices
    train_indices = []
    val_indices = []
    
    for label, indices in class_indices.items():
        train_indices.extend(indices[:train_per_class])
        val_indices.extend(indices[train_per_class:train_per_class+val_per_class])
    
    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset

# Function to train and evaluate models on a specific dataset
def train_models_on_dataset(dataset_path, dataset_name, num_epochs=20):
    try:
        print(f"\n{'='*50}")
        print(f"Training models on {dataset_name} dataset: {dataset_path}")
        print(f"{'='*50}")
        
        # Dataset loading
        dataset_resnet = datasets.ImageFolder(root=dataset_path, transform=transform_full)
        dataset_densenet = datasets.ImageFolder(root=dataset_path, transform=transform_full)

        total_size = len(dataset_resnet)
        print(f"Total images: {total_size}")
        print(f"Classes: {dataset_resnet.classes}")

        # Create stratified splits - 25,000 from each class for training, 5,000 from each class for validation
        train_dataset_resnet, val_dataset_resnet = stratified_split(dataset_resnet)
        train_dataset_densenet, val_dataset_densenet = stratified_split(dataset_densenet)

        print(f"Training set size: {len(train_dataset_resnet)}")
        print(f"Validation set size: {len(val_dataset_resnet)}")

        # DataLoaders
        batch_size = 16  # Increased from 8 to better utilize GPU
        num_workers = 4  # Use multiple workers for faster data loading
        
        train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader_resnet = DataLoader(val_dataset_resnet, batch_size=batch_size, 
                                      shuffle=False, num_workers=num_workers, pin_memory=True)

        train_loader_densenet = DataLoader(train_dataset_densenet, batch_size=batch_size, 
                                         shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader_densenet = DataLoader(val_dataset_densenet, batch_size=batch_size, 
                                       shuffle=False, num_workers=num_workers, pin_memory=True)

        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Train ResNet50
        print(f"\nTraining ResNet50 on {dataset_name}...")
        model_resnet = models.resnet50(weights=None)
        model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 1)
        model_resnet = model_resnet.to(device)
        resnet_history = train_model(model_resnet, train_loader_resnet, val_loader_resnet,
                                     num_epochs, device, f"ResNet50-{dataset_name}", 
                                     f"dna_image_saved_models/resnet50_{dataset_name}_{timestamp}.pth",
                                     f"dna_image_saved_models/resnet50_{dataset_name}_{timestamp}_history.csv")

        # Train DenseNet121
        print(f"\nTraining DenseNet121 on {dataset_name}...")
        model_densenet = models.densenet121(weights=None)
        model_densenet.classifier = nn.Linear(model_densenet.classifier.in_features, 1)
        model_densenet = model_densenet.to(device)
        densenet_history = train_model(model_densenet, train_loader_densenet, val_loader_densenet,
                                       num_epochs, device, f"DenseNet121-{dataset_name}", 
                                       f"dna_image_saved_models/densenet121_{dataset_name}_{timestamp}.pth",
                                       f"dna_image_saved_models/densenet121_{dataset_name}_{timestamp}_history.csv")

        return resnet_history[3], densenet_history[3], timestamp
    except Exception as e:
        print(f"Error training models on {dataset_name}: {str(e)}")
        return None, None, None

# Training function
def train_model(model, train_loader, val_loader, num_epochs, device, model_name, model_save_path, history_save_path):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0  # Track best validation accuracy
    start_time = time.time()

    # Create directory for plots
    plots_dir = "dna_image_training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get plot name from model save path
    plot_base_name = os.path.basename(model_save_path).replace('.pth', '')

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct_train += (predicted.squeeze() == labels.float()).sum().item()
            total_train += labels.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / total_train
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        
        # Use tqdm for progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                running_val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct_val += (predicted.squeeze() == labels.float()).sum().item()
                total_val += labels.size(0)
                
                # Update progress bar
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss = running_val_loss / total_val
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save current epoch results to CSV (append mode)
        os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
        epoch_results_exist = os.path.exists(history_save_path)
        with open(history_save_path, mode='a' if epoch_results_exist else 'w', newline='') as file:
            writer = csv.writer(file)
            if not epoch_results_exist:
                writer.writerow(["Epoch", "Train Loss", "Train Accuracy (%)", "Validation Loss", "Validation Accuracy (%)"])
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model improved! {model_name} model saved as {model_save_path}")
            
        # Plot and save training curves after each epoch
        epochs = list(range(1, epoch + 2))
        # Create figure with 2 subplots
        plt.figure(figsize=(12, 10))
        
        # Loss plot
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title(f'{model_name} Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        plt.title(f'{model_name} Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = f"{plots_dir}/{plot_base_name}_epoch_{epoch+1}.png"
        plt.savefig(plot_path)
        plt.close()
        
        torch.cuda.empty_cache()
        gc.collect()

    training_time = time.time() - start_time

    # Save final training summary
    with open(history_save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Total Training Time (s)", training_time])
        writer.writerow(["Best Validation Accuracy (%)", best_val_acc])
    print(f"Training history saved to {history_save_path}")
    
    # Create and save final training curves plot
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(12, 10))
    
    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} Loss Curves (Final)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy Curves (Final)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    final_plot_path = f"{plots_dir}/{plot_base_name}_final.png"
    plt.savefig(final_plot_path)
    plt.close()
    print(f"Training plots saved to {plots_dir}")

    return train_losses, train_accuracies, val_losses, val_accuracies, training_time, best_val_acc

# Main execution - wrap in try-except to handle interruptions gracefully
if __name__ == "__main__":
    try:
        # Create directory if it doesn't exist
        os.makedirs("dna_image_saved_models", exist_ok=True)
        os.makedirs("dna_image_training_plots", exist_ok=True)

        '''# Train models on all Fixed Color Pattern datasets
        fixed_datasets = [
            ("New_Image_Data/arab_acc", "fixed_arab_acc"),
            ("New_Image_Data/arab_don", "fixed_arab_don"),
            ("New_Image_Data/homo_acc", "fixed_homo_acc"),
            ("New_Image_Data/homo_don", "fixed_homo_don")
        ]

        fixed_results = {}
        for dataset_path, dataset_name in fixed_datasets:
            print(f"\nTraining models on Fixed Color Pattern dataset: {dataset_name}...")
            resnet_acc, densenet_acc, timestamp = train_models_on_dataset(
                dataset_path, dataset_name, num_epochs=20)
            fixed_results[dataset_name] = {
                "resnet": resnet_acc,
                "densenet": densenet_acc,
                "timestamp": timestamp
            }'''

        # Train models on all FCGR datasets
        fcgr_datasets = [
            ("dna_image_fcgr/arab_acc", "fcgr_arab_acc"),
            ("dna_image_fcgr/arab_don", "fcgr_arab_don"),
            ("dna_image_fcgr/homo_acc", "fcgr_homo_acc"),
            ("dna_image_fcgr/homo_don", "fcgr_homo_don")
        ]

        fcgr_results = {}
        for dataset_path, dataset_name in fcgr_datasets:
            print(f"\nTraining models on FCGR dataset: {dataset_name}...")
            resnet_acc, densenet_acc, timestamp = train_models_on_dataset(
                dataset_path, dataset_name, num_epochs=20)
            fcgr_results[dataset_name] = {
                "resnet": resnet_acc,
                "densenet": densenet_acc,
                "timestamp": timestamp
            }

        '''# Print training results summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE - RESULTS SUMMARY")
        print("="*70)

        # Print Fixed Color Pattern results
        print("\nFixed Color Pattern Datasets:")
        print("-"*50)
        print(f"{'Dataset':<20} {'ResNet50 Val Acc':<20} {'DenseNet121 Val Acc':<20}")
        print("-"*50)
        for dataset_name, results in fixed_results.items():
            if results["resnet"] is not None and results["densenet"] is not None:
                print(f"{dataset_name:<20} {results['resnet']:>15.2f}% {results['densenet']:>20.2f}%")
            else:
                print(f"{dataset_name:<20} {'Training Failed':<20} {'Training Failed':<20}")'''

        # Print FCGR results
        print("\nFCGR Datasets:")
        print("-"*50)
        print(f"{'Dataset':<20} {'ResNet50 Val Acc':<20} {'DenseNet121 Val Acc':<20}")
        print("-"*50)
        for dataset_name, results in fcgr_results.items():
            if results["resnet"] is not None and results["densenet"] is not None:
                print(f"{dataset_name:<20} {results['resnet']:>15.2f}% {results['densenet']:>20.2f}%")
            else:
                print(f"{dataset_name:<20} {'Training Failed':<20} {'Training Failed':<20}")

        print(f"\nAll models saved in dna_image_saved_models directory")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current state and exiting...")
    except Exception as e:
        print(f"\n\nAn error occurred during training: {str(e)}")
    finally:
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        print("Training script completed.")
