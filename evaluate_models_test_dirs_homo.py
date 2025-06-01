import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, balanced_accuracy_score,
    cohen_kappa_score
)
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Custom specificity metric
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def load_model(model_path, model_type, device):
    if model_type == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Single output for binary classification
    else:  # densenet121
        from torchvision.models import densenet121
        model = densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 1)  # Single output for binary classification
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()  # Use sigmoid for binary classification
            
            # Handle batch size of 1 case
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs > 0.5).float()  # Threshold at 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'Specificity': specificity_score(y_true, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['AUC'] = auc(fpr, tpr)
    
    return metrics, fpr, tpr

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_homo_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_roc_curve(fpr, tpr, auc_value, title):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend()
    plt.savefig(f'roc_curve_homo_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_metrics_comparison(metrics_dict):
    metrics = list(metrics_dict.keys())
    models = list(metrics_dict[metrics[0]].keys())
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35 / len(models)
    
    for i, model in enumerate(models):
        values = [metrics_dict[metric][model] for metric in metrics]
        plt.bar(x + i*width, values, width, label=model)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison - Homo Sapiens')
    plt.xticks(x + width*len(models)/2, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_comparison_homo_test_set.png')
    plt.close()

def main():
    set_seed(42)
    
    # Select appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # Data transforms - exactly matching training script
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load test datasets from the dedicated test directories
    print("\nLoading Homo sapiens test datasets...")
    test_fcgr_homo_don_dataset = datasets.ImageFolder('Test_Image_fcgr/homo_don', transform=transform)
    test_fcgr_homo_acc_dataset = datasets.ImageFolder('Test_Image_fcgr/homo_acc', transform=transform)
    test_fixed_homo_don_dataset = datasets.ImageFolder('Test_Image_fixed/homo_don', transform=transform)
    test_fixed_homo_acc_dataset = datasets.ImageFolder('Test_Image_fixed/homo_acc', transform=transform)
    
    print(f"\nTest dataset sizes:")
    print(f"FCGR Homo Don test dataset: {len(test_fcgr_homo_don_dataset)} samples")
    print(f"FCGR Homo Acc test dataset: {len(test_fcgr_homo_acc_dataset)} samples")
    print(f"Fixed Color Homo Don test dataset: {len(test_fixed_homo_don_dataset)} samples")
    print(f"Fixed Color Homo Acc test dataset: {len(test_fixed_homo_acc_dataset)} samples")
    
    # Create class distribution counters to understand the dataset
    fcgr_homo_don_class_counts = {0: 0, 1: 0}
    fcgr_homo_acc_class_counts = {0: 0, 1: 0}
    fixed_homo_don_class_counts = {0: 0, 1: 0}
    fixed_homo_acc_class_counts = {0: 0, 1: 0}
    
    # Count samples per class
    for _, label in test_fcgr_homo_don_dataset.samples:
        fcgr_homo_don_class_counts[label] += 1
        
    for _, label in test_fcgr_homo_acc_dataset.samples:
        fcgr_homo_acc_class_counts[label] += 1
        
    for _, label in test_fixed_homo_don_dataset.samples:
        fixed_homo_don_class_counts[label] += 1
        
    for _, label in test_fixed_homo_acc_dataset.samples:
        fixed_homo_acc_class_counts[label] += 1
    
    print(f"\nFCGR Homo Don test dataset class distribution:")
    print(f"  Negative (class 0): {fcgr_homo_don_class_counts[0]} samples")
    print(f"  Positive (class 1): {fcgr_homo_don_class_counts[1]} samples")
    
    print(f"\nFCGR Homo Acc test dataset class distribution:")
    print(f"  Negative (class 0): {fcgr_homo_acc_class_counts[0]} samples")
    print(f"  Positive (class 1): {fcgr_homo_acc_class_counts[1]} samples")
    
    print(f"\nFixed Color Homo Don test dataset class distribution:")
    print(f"  Negative (class 0): {fixed_homo_don_class_counts[0]} samples")
    print(f"  Positive (class 1): {fixed_homo_don_class_counts[1]} samples")
    
    print(f"\nFixed Color Homo Acc test dataset class distribution:")
    print(f"  Negative (class 0): {fixed_homo_acc_class_counts[0]} samples")
    print(f"  Positive (class 1): {fixed_homo_acc_class_counts[1]} samples")
    
    # Use same batch size as training
    batch_size = 8
    fcgr_homo_don_test_loader = DataLoader(test_fcgr_homo_don_dataset, batch_size=batch_size, shuffle=False)
    fcgr_homo_acc_test_loader = DataLoader(test_fcgr_homo_acc_dataset, batch_size=batch_size, shuffle=False)
    fixed_homo_don_test_loader = DataLoader(test_fixed_homo_don_dataset, batch_size=batch_size, shuffle=False)
    fixed_homo_acc_test_loader = DataLoader(test_fixed_homo_acc_dataset, batch_size=batch_size, shuffle=False)
    
    # Load saved models for Homo sapiens
    print("\nLoading Homo sapiens models...")
    models = {
        'FCGR Homo Don': {
            'ResNet50': load_model('dna_image_saved_models/resnet50_fcgr_homo_don_20250416_032643.pth', 'resnet50', device),
            'DenseNet121': load_model('dna_image_saved_models/densenet121_fcgr_homo_don_20250416_032643.pth', 'densenet121', device)
        },
        'FCGR Homo Acc': {
            'ResNet50': load_model('dna_image_saved_models/resnet50_fcgr_homo_acc_20250416_003324.pth', 'resnet50', device),
            'DenseNet121': load_model('dna_image_saved_models/densenet121_fcgr_homo_acc_20250416_003324.pth', 'densenet121', device)
        },
        'Fixed Homo Don': {
            'ResNet50': load_model('dna_image_saved_models/resnet50_fixed_homo_don_20250416_151039.pth', 'resnet50', device),
            'DenseNet121': load_model('dna_image_saved_models/densenet121_fixed_homo_don_20250416_151039.pth', 'densenet121', device)
        },
        'Fixed Homo Acc': {
            'ResNet50': load_model('dna_image_saved_models/resnet50_fixed_homo_acc_20250416_043420.pth', 'resnet50', device),
            'DenseNet121': load_model('dna_image_saved_models/densenet121_fixed_homo_acc_20250416_043420.pth', 'densenet121', device)
        }
    }
    
    # Evaluate models and collect metrics
    all_metrics = {}
    # Map dataset types to their respective test loaders
    test_loaders = {
        'FCGR Homo Don': fcgr_homo_don_test_loader,
        'FCGR Homo Acc': fcgr_homo_acc_test_loader,
        'Fixed Homo Don': fixed_homo_don_test_loader,
        'Fixed Homo Acc': fixed_homo_acc_test_loader
    }
    
    for dataset_type, dataset_models in models.items():
        test_loader = test_loaders[dataset_type]
        
        for model_name, model in dataset_models.items():
            print(f"\nEvaluating {model_name} on {dataset_type} dataset...")
            
            y_pred, y_true, y_prob = evaluate_model(model, test_loader, device)
            print(f"Predictions shape: {y_pred.shape}")
            print(f"Labels shape: {y_true.shape}")
            print(f"Unique predictions: {np.unique(y_pred, return_counts=True)}")
            print(f"Unique labels: {np.unique(y_true, return_counts=True)}")
            
            metrics, fpr, tpr = calculate_metrics(y_true, y_pred, y_prob)
            print("\nMetrics:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Store metrics
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = {}
                all_metrics[metric_name][f"{dataset_type}_{model_name}"] = value
            
            # Plot confusion matrix
            plot_confusion_matrix(y_true, y_pred, f"{dataset_type} - {model_name}")
            
            # Plot ROC curve
            plot_roc_curve(fpr, tpr, metrics['AUC'], f"{dataset_type} - {model_name}")
    
    # Plot metrics comparison
    plot_metrics_comparison(all_metrics)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv('dedicated_test_set_metrics_homo.csv')
    
    print("\nEvaluation complete! Results saved to:")
    print("1. dedicated_test_set_metrics_homo.csv - All metrics for each model")
    print("2. confusion_matrix_homo_*.png - Confusion matrices for each model")
    print("3. roc_curve_homo_*.png - ROC curves for each model")
    print("4. metrics_comparison_homo_test_set.png - Bar chart comparing all metrics across models")

if __name__ == "__main__":
    main() 