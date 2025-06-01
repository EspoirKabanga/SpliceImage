import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
import random
import re
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import pandas as pd
import csv
from torch.utils.data import DataLoader

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

set_seed()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Create output directories
os.makedirs("final_figures", exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target = output
        else:
            target = output[0, target_class]
        self.model.zero_grad()
        target.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam[0, 0].cpu().numpy()

def load_model(model_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    target_layer = model.layer4[-1]
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, target_layer

def generate_saliency_map(model, input_tensor):
    """Generate saliency map for the input tensor"""
    input_tensor.requires_grad = True
    output = model(input_tensor)
    model.zero_grad()
    output.backward()
    saliency = input_tensor.grad.abs()
    saliency = torch.max(saliency, dim=1)[0]
    
    # Normalize saliency map
    saliency_np = saliency[0].cpu().numpy()
    saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
    return saliency_np

def get_nucleotide_index(nucleotide):
    """Map nucleotides to indices for FCGR positioning (same as in dna_image_convert_FCGR.py)."""
    if nucleotide == 'A':
        return 0
    elif nucleotide == 'C':
        return 1
    elif nucleotide == 'G':
        return 2
    elif nucleotide == 'T':
        return 3
    else:  # Handle ambiguous nucleotides like 'N'
        return -1

def map_kmer_to_fcgr_position(kmer, image_size=64):
    """Map a k-mer to its position in the FCGR image."""
    k = len(kmer)
    x, y = 0, 0
    
    for j, nuc in enumerate(kmer):
        nuc_idx = get_nucleotide_index(nuc)
        if nuc_idx == -1:  # Skip if ambiguous nucleotide
            return None
            
        # Update position based on binary representation
        bit_x = (nuc_idx & 1)  # Least significant bit
        bit_y = ((nuc_idx >> 1) & 1)  # Most significant bit
        
        # Update coordinates (each nucleotide contributes to position)
        x += bit_x * (2**(k-j-1))
        y += bit_y * (2**(k-j-1))
    
    # Scale to image size
    scale_factor = image_size / (2**k)
    region_size = max(1, int(scale_factor))
    
    x_scaled = int(x * scale_factor)
    y_scaled = int(y * scale_factor)
    
    return (x_scaled, y_scaled, x_scaled + region_size, y_scaled + region_size)

def get_k_mers_from_sequence(sequence, k=6):
    """Extract all k-mers from a DNA sequence."""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        # Only include k-mers with standard nucleotides
        if all(nuc in 'ACGT' for nuc in kmer):
            kmers.append(kmer)
    return kmers

def calculate_kmer_importances(heatmap, kmer_regions):
    """Calculate importance of each k-mer based on the average heatmap value in its region"""
    kmer_importances = {}
    
    for kmer, (x_min, y_min, x_max, y_max) in kmer_regions.items():
        # Extract the region from the heatmap
        region = heatmap[y_min:y_max, x_min:x_max]
        
        # Calculate the average value in the region
        avg_value = np.mean(region)
        kmer_importances[kmer] = avg_value
    
    return kmer_importances

def count_kmers_in_sequence(sequence, k=6):
    """Count k-mers in a sequence"""
    kmer_counts = Counter()
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(nuc in 'ACGT' for nuc in kmer):
            kmer_counts[kmer] += 1
    
    return kmer_counts

def extract_sequence_id(image_path):
    """Extract the sequence ID (number) from an image path."""
    filename = os.path.basename(image_path)
    if filename.startswith('seq_') and filename.endswith('.png'):
        try:
            seq_id = int(filename.replace('seq_', '').replace('.png', ''))
            return seq_id
        except ValueError:
            return None
    return None

def load_dna_sequence(seq_id, dataset_name):
    """Load the actual DNA sequence for a given sequence ID."""
    # Map dataset names to sequence files
    dataset_files = {
        'arab_don_pos': 'DRANet/arabidopsis_donor_positive.txt',
        'arab_don_neg': 'DRANet/arabidopsis_donor_negative.txt',
        'arab_acc_pos': 'DRANet/arabidopsis_acceptor_positive.txt',
        'arab_acc_neg': 'DRANet/arabidopsis_acceptor_negative.txt',
        'homo_don_pos': 'DRANet/homo_donor_positive.txt',
        'homo_don_neg': 'DRANet/homo_donor_negative.txt',
        'homo_acc_pos': 'DRANet/homo_acceptor_positive.txt',
        'homo_acc_neg': 'DRANet/homo_acceptor_negative.txt',
    }
    
    if dataset_name not in dataset_files:
        print(f"Warning: Unknown dataset {dataset_name}, using synthetic sequence")
        return generate_synthetic_sequence(str(seq_id), length=100)
    
    # Get the file path
    file_path = dataset_files[dataset_name]
    
    try:
        # Read the sequence at the specified line
        with open(file_path, 'r') as f:
            for i, line in enumerate(f, 1):
                if i == seq_id:
                    return line.strip()
        
        # If we couldn't find the sequence, generate a synthetic one
        print(f"Warning: Sequence {seq_id} not found in {file_path}, using synthetic sequence")
        return generate_synthetic_sequence(str(seq_id), length=100)
    except Exception as e:
        print(f"Error loading sequence {seq_id} from {file_path}: {str(e)}")
        return generate_synthetic_sequence(str(seq_id), length=100)

def generate_synthetic_sequence(seq_id, length=100):
    """Generate a synthetic DNA sequence from an ID (for testing)"""
    # Use seq_id as a seed for reproducibility
    np.random.seed(sum(ord(c) for c in seq_id))
    
    nucleotides = ['A', 'C', 'G', 'T']
    sequence = ''.join(np.random.choice(nucleotides) for _ in range(length))
    return sequence

def visualize_fcgr_with_kmers(image, heatmap, saliency, kmer_regions, kmer_importances, saliency_importances, k=6, output_path=None, sequence_name=None):
    """Visualize FCGR image with k-mer regions overlaid and colored by importance"""
    # Normalize the image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 0.5) + 0.5  # Denormalize
    
    # Create figure with subplots - 1 row, 3 columns with proper spacing
    # 5cm ≈ 2 inches of separation
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    
    # Add spacing between subplots (2 inches ≈ 5cm)
    plt.subplots_adjust(wspace=0.3, hspace=0)
    
    # Increased font sizes
    plt.rcParams.update({'font.size': 16})
    title_fontsize = 24  # Increased from 18 to 24
    kmer_fontsize = 16
    
    # Original image
    axs[0].imshow(image)
    # Use sequence name as the title for the first image if provided, otherwise use default
    axs[0].set_title(sequence_name if sequence_name else "Original FCGR Image", fontsize=title_fontsize)
    # Completely remove axes and frame
    axs[0].axis('off')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    # Sort k-mers by GradCAM importance and Saliency importance separately
    sorted_gradcam_kmers = sorted(kmer_importances.items(), key=lambda x: x[1], reverse=True)
    sorted_saliency_kmers = sorted(saliency_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Display only top 10 k-mers
    top_n = 10
    top_gradcam_kmers = sorted_gradcam_kmers[:top_n]
    top_saliency_kmers = sorted_saliency_kmers[:top_n]
    
    # Grad-CAM heatmap with k-mer overlay
    axs[1].imshow(image)  # Use original image as base
    axs[1].imshow(heatmap, cmap='jet', alpha=0.7)  # Overlay Grad-CAM with transparency
    axs[1].set_title("Grad-CAM Heatmap with Top 10 6-mers", fontsize=title_fontsize)
    # Completely remove axes and frame
    axs[1].axis('off')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    # Add k-mer rectangles to GradCAM
    for kmer, importance in top_gradcam_kmers:
        if kmer in kmer_regions:
            x_min, y_min, x_max, y_max = kmer_regions[kmer]
            width = x_max - x_min
            height = y_max - y_min
            
            # For GradCAM visualization
            rect = patches.Rectangle((x_min, y_min), width, height, 
                                    linewidth=2, edgecolor='cyan', 
                                    facecolor='none')
            axs[1].add_patch(rect)
            
            # Add k-mer text on GradCAM
            axs[1].text(x_min + width/2, y_min + height/2, kmer, 
                          ha='center', va='center', 
                          fontsize=kmer_fontsize, color='cyan', 
                          fontweight='bold',
                          bbox=dict(facecolor='black', alpha=0.7, pad=1))
    
    # Saliency map with k-mer overlay
    axs[2].imshow(image)  # Use original image as base
    axs[2].imshow(saliency, cmap='hot', alpha=0.7)  # Overlay Saliency with transparency
    axs[2].set_title("Saliency Map with Top 10 6-mers", fontsize=title_fontsize)
    # Completely remove axes and frame
    axs[2].axis('off')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    
    # Add k-mer rectangles to Saliency map - using the saliency-based importance
    for kmer, importance in top_saliency_kmers:
        if kmer in kmer_regions:
            x_min, y_min, x_max, y_max = kmer_regions[kmer]
            width = x_max - x_min
            height = y_max - y_min
            
            # For Saliency visualization
            rect = patches.Rectangle((x_min, y_min), width, height, 
                                    linewidth=2, edgecolor='yellow', 
                                    facecolor='none')
            axs[2].add_patch(rect)
            
            # Add k-mer text on Saliency
            axs[2].text(x_min + width/2, y_min + height/2, kmer, 
                          ha='center', va='center', 
                          fontsize=kmer_fontsize, color='yellow', 
                          fontweight='bold',
                          bbox=dict(facecolor='black', alpha=0.7, pad=1))
    
    # Adjust layout with adequate spacing
    plt.tight_layout()
    
    # Save the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    return fig

def create_combined_visualization(vis_data, species, dataset_type):
    """Create a combined visualization with two TP sequences, one on top of the other"""
    if len(vis_data) == 0:
        print(f"No visualization data for {species} {dataset_type}")
        return None
    
    # Number of visualizations (typically 2)
    num_vis = len(vis_data)
    
    # Create a figure with proper spacing
    # For each visualization: 8 inches height + 2 inches (5cm) spacing between rows
    fig_height = num_vis * 8 + (num_vis - 1) * 2
    fig = plt.figure(figsize=(24, fig_height))
    
    # Set proper spacing between subplots (2 inches ≈ 5cm)
    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    
    # Increased font sizes
    plt.rcParams.update({'font.size': 16})
    title_fontsize = 24  # Increased from 18 to 24
    kmer_fontsize = 16
    
    # Process each visualization
    for i, (image, heatmap, saliency, kmer_regions, kmer_importances, saliency_importances, sequence_name) in enumerate(vis_data):
        # Create a row with 3 columns for this TP
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = (image * 0.5) + 0.5  # Denormalize
        
        # Create a subplot within this row - we need to calculate positions to ensure proper spacing
        row_axes = [
            plt.subplot(num_vis, 3, i*3+1),
            plt.subplot(num_vis, 3, i*3+2),
            plt.subplot(num_vis, 3, i*3+3)
        ]
        
        # Original image
        row_axes[0].imshow(image)
        row_axes[0].set_title(sequence_name, fontsize=title_fontsize)
        # Completely remove axes and frame
        row_axes[0].axis('off')
        row_axes[0].set_xticks([])
        row_axes[0].set_yticks([])
        
        # Sort k-mers by importance
        sorted_gradcam_kmers = sorted(kmer_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        sorted_saliency_kmers = sorted(saliency_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Grad-CAM heatmap
        row_axes[1].imshow(image)  # Use original image as base
        row_axes[1].imshow(heatmap, cmap='jet', alpha=0.7)  # Overlay Grad-CAM with transparency
        row_axes[1].set_title("Grad-CAM Heatmap with Top 10 6-mers", fontsize=title_fontsize)
        # Completely remove axes and frame
        row_axes[1].axis('off')
        row_axes[1].set_xticks([])
        row_axes[1].set_yticks([])
        
        # Add k-mer rectangles to GradCAM
        for kmer, importance in sorted_gradcam_kmers:
            if kmer in kmer_regions:
                x_min, y_min, x_max, y_max = kmer_regions[kmer]
                width = x_max - x_min
                height = y_max - y_min
                
                # For GradCAM visualization
                rect = patches.Rectangle((x_min, y_min), width, height, 
                                        linewidth=2, edgecolor='cyan', 
                                        facecolor='none')
                row_axes[1].add_patch(rect)
                
                # Add k-mer text on GradCAM
                row_axes[1].text(x_min + width/2, y_min + height/2, kmer, 
                               ha='center', va='center', 
                               fontsize=kmer_fontsize, color='cyan', 
                               fontweight='bold',
                               bbox=dict(facecolor='black', alpha=0.7, pad=1))
        
        # Saliency map
        row_axes[2].imshow(image)  # Use original image as base
        row_axes[2].imshow(saliency, cmap='hot', alpha=0.7)  # Overlay Saliency with transparency
        row_axes[2].set_title("Saliency Map with Top 10 6-mers", fontsize=title_fontsize)
        # Completely remove axes and frame
        row_axes[2].axis('off')
        row_axes[2].set_xticks([])
        row_axes[2].set_yticks([])
        
        # Add k-mer rectangles to Saliency map
        for kmer, importance in sorted_saliency_kmers:
            if kmer in kmer_regions:
                x_min, y_min, x_max, y_max = kmer_regions[kmer]
                width = x_max - x_min
                height = y_max - y_min
                
                # For Saliency visualization
                rect = patches.Rectangle((x_min, y_min), width, height, 
                                        linewidth=2, edgecolor='yellow', 
                                        facecolor='none')
                row_axes[2].add_patch(rect)
                
                # Add k-mer text on Saliency
                row_axes[2].text(x_min + width/2, y_min + height/2, kmer, 
                               ha='center', va='center', 
                               fontsize=kmer_fontsize, color='yellow', 
                               fontweight='bold',
                               bbox=dict(facecolor='black', alpha=0.7, pad=1))
    
    # Adjust layout with adequate spacing
    plt.tight_layout()
    
    # Save the figure
    output_path = f"final_figures/{species}_{dataset_type}_combined_visualizations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved to {output_path}")
    
    return fig

def evaluate_model(model, test_loader, device):
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []
    all_filenames = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()  # For binary classification
            preds = (probs > 0.5).float()  # Threshold at 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.cpu())
            
            # Get the image paths for this batch
            start_idx = batch_idx * test_loader.batch_size
            for i in range(len(images)):
                img_idx = start_idx + i
                if img_idx < len(test_loader.dataset.imgs):
                    all_filenames.append(test_loader.dataset.imgs[img_idx][0])
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images, all_filenames

def get_tp_sequences(predictions, labels, images, filenames, num_samples=2):
    """Get the first num_samples True Positive sequences"""
    tp_indices = np.where((predictions == 1) & (labels == 1))[0]
    
    tp_samples = []
    if len(tp_indices) > 0:
        selected_tp = tp_indices[:min(num_samples, len(tp_indices))]
        tp_samples = [(images[i], filenames[i]) for i in selected_tp]
    
    return tp_samples

def extract_class_from_path(file_path):
    """Extract class label (pos/neg) from file path"""
    if '/pos/' in file_path:
        return 'pos'
    elif '/neg/' in file_path:
        return 'neg'
    else:
        return 'unknown'

def analyze_species_dataset(species, dataset_type, model_path):
    """Analyze a specific species dataset"""
    print(f"\nProcessing {species} {dataset_type} dataset...")
    
    # Load dataset
    dataset_path = f"Test_Image_fcgr/{species}_{dataset_type}"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} not found!")
        return None
    
    # Load model
    try:
        model, target_layer = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Load dataset and create test loader
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Evaluate model
    print(f"Evaluating model on {species} {dataset_type} dataset...")
    predictions, labels, probs, images, filenames = evaluate_model(model, test_loader, device)
    
    # Get performance metrics
    accuracy = np.mean(predictions == labels)
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    print(f"Model performance on {species} {dataset_type}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  True Positives: {tp}, True Negatives: {tn}")
    print(f"  False Positives: {fp}, False Negatives: {fn}")
    
    # Get first 10 True Positive samples for CSV data
    tp_indices = np.where((predictions == 1) & (labels == 1))[0]
    
    if len(tp_indices) == 0:
        print(f"No True Positive samples found for {species} {dataset_type}")
        return None
    
    # Get 10 TP samples for CSV data
    num_tp_for_csv = min(10, len(tp_indices))
    tp_indices_for_csv = tp_indices[:num_tp_for_csv]
    
    # Get 2 TP samples for visualization
    num_tp_for_viz = min(2, len(tp_indices))
    tp_indices_for_viz = tp_indices[:num_tp_for_viz]
    
    # Create lists to store top 10 6-mers from all TP samples
    all_gradcam_kmers = []
    all_saliency_kmers = []
    
    # Store visualization data for combined figure
    visualization_data = []
    
    # Process each TP sample for CSV data
    for i, idx in enumerate(tp_indices_for_csv):
        image = images[idx]
        filename = filenames[idx]
        
        seq_id = extract_sequence_id(filename)
        if seq_id is None:
            print(f"Could not extract sequence ID from {filename}")
            continue
        
        print(f"Processing sequence {seq_id} from {filename}")
        
        # Determine dataset name from path
        dataset_name = None
        if f"{species}_{dataset_type}/pos" in filename:
            dataset_name = f"{species}_{dataset_type}_pos"
        elif f"{species}_{dataset_type}/neg" in filename:
            dataset_name = f"{species}_{dataset_type}_neg"
        
        if dataset_name is None:
            print(f"Could not determine dataset name from {filename}")
            continue
        
        # Prepare for analysis
        input_tensor = image.unsqueeze(0).to(device)
        
        # Set up GradCAM
        grad_cam = GradCAM(model, target_layer)
        
        # Generate GradCAM and Saliency Map
        gradcam_heatmap = grad_cam.generate_cam(input_tensor)
        saliency_map = generate_saliency_map(model, input_tensor.clone())
        
        # Get image as numpy array
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 0.5) + 0.5  # Denormalize
        image_size = min(image_np.shape[0], image_np.shape[1])
        
        # Load the actual DNA sequence
        sequence = load_dna_sequence(seq_id, dataset_name)
        
        # Extract 6-mers from sequence
        kmers = get_k_mers_from_sequence(sequence, k=6)
        
        # Map each 6-mer to its position in the FCGR image
        kmer_regions = {}
        for kmer in kmers:
            position = map_kmer_to_fcgr_position(kmer, image_size=image_size)
            if position:
                kmer_regions[kmer] = position
        
        # Calculate importance of each 6-mer
        gradcam_importances = calculate_kmer_importances(gradcam_heatmap, kmer_regions)
        saliency_importances = calculate_kmer_importances(saliency_map, kmer_regions)
        
        # Count frequency of each 6-mer in the sequence
        kmer_counts = count_kmers_in_sequence(sequence, k=6)
        
        # Get top 10 6-mers and store them for the combined CSV files
        top_gradcam_kmers = sorted(gradcam_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        top_saliency_kmers = sorted(saliency_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store the top k-mers with sequence ID, importance, and count
        for kmer, importance in top_gradcam_kmers:
            count = kmer_counts[kmer]
            all_gradcam_kmers.append((species, dataset_type, seq_id, kmer, importance, count))
        
        for kmer, importance in top_saliency_kmers:
            count = kmer_counts[kmer]
            all_saliency_kmers.append((species, dataset_type, seq_id, kmer, importance, count))
        
        # For the first 2 TP samples, also generate individual visualizations and collect data for combined visualization
        if idx in tp_indices_for_viz:
            # Create output directory
            os.makedirs("final_figures", exist_ok=True)
            
            # Get file basename and class
            file_basename = os.path.basename(filename)
            file_class = extract_class_from_path(filename)
            
            # Create sequence name with format "seq_XXXXX.png (Class: pos)"
            sequence_title = f"{file_basename} (Class: {file_class})"
            
            # Store visualization data for combined figure
            visualization_data.append((
                image_np,
                gradcam_heatmap,
                saliency_map,
                kmer_regions,
                gradcam_importances,
                saliency_importances,
                sequence_title
            ))
            
            # Generate individual visualization 
            output_path = f"final_figures/{species}_{dataset_type}_seq_{seq_id}.png"
            
            visualize_fcgr_with_kmers(
                image=image_np, 
                heatmap=gradcam_heatmap, 
                saliency=saliency_map,
                kmer_regions=kmer_regions, 
                kmer_importances=gradcam_importances,
                saliency_importances=saliency_importances,
                k=6,
                output_path=output_path,
                sequence_name=sequence_title
            )
    
    # Create combined visualization with all TP samples
    if len(visualization_data) > 0:
        create_combined_visualization(visualization_data, species, dataset_type)
    
    # Save combined GradCAM results to CSV
    gradcam_csv = f"final_figures/{species}_{dataset_type}_gradcam_top10_all.csv"
    with open(gradcam_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Species", "Dataset", "Sequence ID", "6-mer", "GradCAM Importance", "Count in Sequence"])
        for species_name, dataset_name, seq_id, kmer, importance, count in all_gradcam_kmers:
            writer.writerow([species_name, dataset_name, seq_id, kmer, f"{importance:.6f}", count])
    
    # Save combined Saliency results to CSV
    saliency_csv = f"final_figures/{species}_{dataset_type}_saliency_top10_all.csv"
    with open(saliency_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Species", "Dataset", "Sequence ID", "6-mer", "Saliency Importance", "Count in Sequence"])
        for species_name, dataset_name, seq_id, kmer, importance, count in all_saliency_kmers:
            writer.writerow([species_name, dataset_name, seq_id, kmer, f"{importance:.6f}", count])
    
    return tp_indices_for_viz

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs("final_figures", exist_ok=True)
    
    # Define species and dataset types
    species_list = ["arab", "homo"]
    dataset_types = ["acc", "don"]
    
    # Create combined CSV files for all results
    gradcam_csv_all = "final_figures/all_gradcam_top10.csv"
    saliency_csv_all = "final_figures/all_saliency_top10.csv"
    
    # Create the CSV files with headers
    with open(gradcam_csv_all, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Species", "Dataset", "Sequence ID", "6-mer", "GradCAM Importance", "Count in Sequence"])
    
    with open(saliency_csv_all, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Species", "Dataset", "Sequence ID", "6-mer", "Saliency Importance", "Count in Sequence"])
    
    # ResNet50 model paths
    model_paths = {
        'arab_acc': 'dna_image_saved_models/resnet50_fcgr_arab_acc_20250415_185111.pth',
        'arab_don': 'dna_image_saved_models/resnet50_fcgr_arab_don_20250415_214118.pth',
        'homo_acc': 'dna_image_saved_models/resnet50_fcgr_homo_acc_20250416_003324.pth',
        'homo_don': 'dna_image_saved_models/resnet50_fcgr_homo_don_20250416_032643.pth'
    }
    
    # Process each species and dataset type
    for species in species_list:
        for dataset_type in dataset_types:
            model_key = f"{species}_{dataset_type}"
            
            if model_key in model_paths:
                model_path = model_paths[model_key]
                
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue
                
                # Process the dataset and get the top k-mers information
                result = analyze_species_dataset(species, dataset_type, model_path)
                
                # Read the individual CSV files and append to the combined ones
                try:
                    gradcam_csv = f"final_figures/{species}_{dataset_type}_gradcam_top10_all.csv"
                    if os.path.exists(gradcam_csv):
                        with open(gradcam_csv, 'r') as src, open(gradcam_csv_all, 'a', newline='') as dest:
                            reader = csv.reader(src)
                            writer = csv.writer(dest)
                            next(reader)  # Skip header
                            for row in reader:
                                writer.writerow(row)
                
                    saliency_csv = f"final_figures/{species}_{dataset_type}_saliency_top10_all.csv"
                    if os.path.exists(saliency_csv):
                        with open(saliency_csv, 'r') as src, open(saliency_csv_all, 'a', newline='') as dest:
                            reader = csv.reader(src)
                            writer = csv.writer(dest)
                            next(reader)  # Skip header
                            for row in reader:
                                writer.writerow(row)
                except Exception as e:
                    print(f"Error appending to combined CSV files: {str(e)}")
            else:
                print(f"No model defined for {model_key}")
    

if __name__ == "__main__":
    main() 