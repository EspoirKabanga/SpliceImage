import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import random
import csv
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.patches as patches
import re  # For regular expression pattern matching

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Biologically significant dinucleotides in splice sites
def get_splice_site_motifs(site_type):
    """
    Returns biologically significant dinucleotides for different splice site types.
    
    Parameters:
    -----------
    site_type : str
        Type of splice site: 'donor', 'acceptor', or 'branch_point'
    
    Returns:
    --------
    dict : Dictionary of dinucleotides and their biological significance
    """
    motifs = {
        'donor': {
            'GT': 'Canonical donor site (5\' splice site)',
            'GC': 'Non-canonical donor site (rare)',
            'AT': 'U12-type donor site (rare)',
            'AG': 'Often precedes GT in donor sites'
        },
        'acceptor': {
            'AG': 'Canonical acceptor site (3\' splice site)',
            'AC': 'U12-type acceptor site (rare)',
            'CT': 'Part of polypyrimidine tract',
            'TC': 'Part of polypyrimidine tract',
            'CC': 'Part of polypyrimidine tract',
            'TT': 'Part of polypyrimidine tract'
        },
        'branch_point': {
            'CT': 'Common in branch point region',
            'AC': 'Part of branch point consensus (YTRAY)',
            'TC': 'Part of branch point consensus (YTRAY)',
            'TA': 'Part of branch point consensus (YTRAY)',
            'CA': 'Contains branch point A nucleotide'
        }
    }
    
    return motifs.get(site_type, {})

def identify_important_dinucleotides(gradcam_heatmap, dinucleotide_grid, threshold=0.7):
    """
    Identifies biologically important dinucleotides based on GradCAM activation.
    
    Parameters:
    -----------
    gradcam_heatmap : np.ndarray
        The GradCAM heatmap
    dinucleotide_grid : np.ndarray
        Grid of dinucleotides
    threshold : float
        Activation threshold for importance
    
    Returns:
    --------
    list : List of tuples (dinucleotide, count, avg_importance)
    """
    # Normalize heatmap if it's not already
    if gradcam_heatmap.max() > 1.0:
        heatmap_norm = (gradcam_heatmap - gradcam_heatmap.min()) / (gradcam_heatmap.max() - gradcam_heatmap.min() + 1e-8)
    else:
        heatmap_norm = gradcam_heatmap
    
    # Get high activation regions
    high_activation = heatmap_norm >= threshold
    
    # Get dinucleotides in high activation regions
    important_dinucs = dinucleotide_grid[high_activation]
    
    # Count occurrences and calculate average importance
    dinuc_stats = {}
    for i, j in zip(*np.where(high_activation)):
        dinuc = dinucleotide_grid[i, j]
        importance = heatmap_norm[i, j]
        
        if dinuc not in dinuc_stats:
            dinuc_stats[dinuc] = {'count': 0, 'importance_sum': 0.0}
        
        dinuc_stats[dinuc]['count'] += 1
        dinuc_stats[dinuc]['importance_sum'] += importance
    
    # Calculate average importance and prepare results
    results = []
    for dinuc, stats in dinuc_stats.items():
        avg_importance = stats['importance_sum'] / stats['count']
        results.append((dinuc, stats['count'], avg_importance))
    
    # Sort by average importance
    return sorted(results, key=lambda x: x[2], reverse=True)

# Define Grad-CAM class - modified to not normalize within the class
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
        cam = F.relu(cam)  # Apply ReLU without normalization
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam[0, 0].cpu().numpy()  # Return unnormalized CAM

def load_model(model_path, model_type):
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        target_layer = model.layer4[-1]
    else:  # densenet121
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        target_layer = model.features.denseblock4.denselayer16
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, target_layer

# Modified to return unnormalized saliency map
def generate_saliency_map(model, input_tensor):
    """Generate saliency map for the input tensor without normalization"""
    input_tensor.requires_grad = True
    output = model(input_tensor)
    model.zero_grad()
    output.backward()
    saliency = input_tensor.grad.abs()
    saliency = torch.max(saliency, dim=1)[0]
    
    # Return unnormalized saliency map
    saliency_np = saliency[0].cpu().numpy()
    return saliency_np

def create_dinucleotide_table(sequence):
    """Create a dinucleotide table for the sequence"""
    sequence_length = len(sequence)
    table = []
    for i in range(sequence_length):
        row = []
        for j in range(sequence_length):
            dinuc = sequence[i] + sequence[j]
            row.append(dinuc)
        table.append(row)
    return np.array(table)

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return predictions, labels, probabilities, images, and paths"""
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()  # Use sigmoid for binary classification
            preds = (probs > 0.5).float()  # Threshold at 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.cpu())
            all_paths.extend(paths)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images, all_paths

def get_classification_type(pred, label):
    """Determine if prediction is TP, TN, FP, or FN"""
    if pred == 1 and label == 1:
        return "TP"
    elif pred == 0 and label == 0:
        return "TN"
    elif pred == 1 and label == 0:
        return "FP"
    else:  # pred == 0 and label == 1
        return "FN"

def visualize_section(sequence_row_part, sequence_col_part, gradcam_part, saliency_part, 
                      gradcam_min, gradcam_max, saliency_min, saliency_max,
                      output_path, section_idx, row, col, 
                      dataset_type=None, is_positive=None):
    """Create visualization for one section of the sequence with consistent normalization"""
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure with two subplots (removing the biological insights panel)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Create dinucleotide table for this section by combining row and column parts
    table_data = []
    for i in range(len(sequence_row_part)):
        row_data = []
        for j in range(len(sequence_col_part)):
            dinuc = sequence_row_part[i] + sequence_col_part[j]
            row_data.append(dinuc)
        table_data.append(row_data)
    dinucleotide_grid = np.array(table_data)
    
    # Normalize GradCAM using global min/max
    gradcam_norm = (gradcam_part - gradcam_min) / (gradcam_max - gradcam_min + 1e-8)
    
    # Plot Grad-CAM
    ax1.set_facecolor('white')
    im1 = ax1.imshow(gradcam_norm, cmap='jet', alpha=0.7, vmin=0, vmax=1)
    
    # Add grid lines to separate cells
    for i in range(len(sequence_row_part)+1):
        ax1.axhline(i-0.5, color='black', linewidth=0.5)
    for j in range(len(sequence_col_part)+1):
        ax1.axvline(j-0.5, color='black', linewidth=0.5)
        
    ax1.axis('off')
    ax1.set_title(f'Grad-CAM R{row}-C{col}', fontsize=16, pad=20)
    
    # Add dinucleotide text over Grad-CAM with better spacing
    fontsize = 5  # Smaller font size to create space between cells
    
    for i in range(len(sequence_row_part)):
        for j in range(len(sequence_col_part)):
            # Determine text color based on heatmap intensity for better visibility
            text_color = 'white' if gradcam_norm[i, j] > 0.5 else 'black'
            
            # Add text without changing background color
            ax1.text(j, i, dinucleotide_grid[i, j], 
                    ha='center', 
                    va='center', 
                    fontsize=fontsize,
                    color=text_color)
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Activation Intensity')
    
    # Normalize Saliency using global min/max
    saliency_norm = (saliency_part - saliency_min) / (saliency_max - saliency_min + 1e-8)
    
    # Plot Saliency Map
    ax2.set_facecolor('white')
    im2 = ax2.imshow(saliency_norm, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    
    # Add grid lines to separate cells
    for i in range(len(sequence_row_part)+1):
        ax2.axhline(i-0.5, color='black', linewidth=0.5)
    for j in range(len(sequence_col_part)+1):
        ax2.axvline(j-0.5, color='black', linewidth=0.5)
        
    ax2.axis('off')
    ax2.set_title(f'Saliency Map R{row}-C{col}', fontsize=16, pad=20)
    
    # Add dinucleotide text over Saliency Map with better spacing
    for i in range(len(sequence_row_part)):
        for j in range(len(sequence_col_part)):
            # Determine text color based on heatmap intensity for better visibility
            text_color = 'white' if saliency_norm[i, j] > 0.5 else 'black'
            
            # Add text without changing background color
            ax2.text(j, i, dinucleotide_grid[i, j], 
                    ha='center', 
                    va='center', 
                    fontsize=fontsize,
                    color=text_color)
    
    # Add colorbar
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Saliency Intensity')
    
    # Add section information at the bottom
    plt.figtext(0.5, 0.01, f"Section {section_idx} (Grid Position R{row}-C{col})", 
                ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Section {section_idx} visualization saved to {output_path}")
    
    # Create separate individual Grad-CAM visualization with dinucleotides but no labels
    fig_gradcam = plt.figure(figsize=(7, 7))
    ax_gradcam = fig_gradcam.add_subplot(111)
    
    # Set the same exact visualization as in the combined figure
    ax_gradcam.set_facecolor('white')
    ax_gradcam.imshow(gradcam_norm, cmap='jet', alpha=0.7, vmin=0, vmax=1)
    
    # Add grid lines to separate cells
    for i in range(len(sequence_row_part)+1):
        ax_gradcam.axhline(i-0.5, color='black', linewidth=0.5)
    for j in range(len(sequence_col_part)+1):
        ax_gradcam.axvline(j-0.5, color='black', linewidth=0.5)
    
    # Add dinucleotide text over Grad-CAM with same formatting
    for i in range(len(sequence_row_part)):
        for j in range(len(sequence_col_part)):
            text_color = 'white' if gradcam_norm[i, j] > 0.5 else 'black'
            
            # Remove highlighting for biologically significant dinucleotides
            # Just add the text without special highlighting
            ax_gradcam.text(j, i, dinucleotide_grid[i, j], 
                          ha='center', 
                          va='center', 
                          fontsize=fontsize,
                          color=text_color)
    
    ax_gradcam.axis('off')
    gradcam_path = output_path.replace('.png', '_gradcam_only.png')
    plt.savefig(gradcam_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create separate individual Saliency visualization with dinucleotides but no labels
    fig_saliency = plt.figure(figsize=(7, 7))
    ax_saliency = fig_saliency.add_subplot(111)
    
    # Set the same exact visualization as in the combined figure
    ax_saliency.set_facecolor('white')
    ax_saliency.imshow(saliency_norm, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    
    # Add grid lines to separate cells
    for i in range(len(sequence_row_part)+1):
        ax_saliency.axhline(i-0.5, color='black', linewidth=0.5)
    for j in range(len(sequence_col_part)+1):
        ax_saliency.axvline(j-0.5, color='black', linewidth=0.5)
    
    # Add dinucleotide text over Saliency with same formatting
    for i in range(len(sequence_row_part)):
        for j in range(len(sequence_col_part)):
            text_color = 'white' if saliency_norm[i, j] > 0.5 else 'black'
            
            # Remove highlighting for biologically significant dinucleotides
            # Just add the text without special highlighting
            ax_saliency.text(j, i, dinucleotide_grid[i, j], 
                           ha='center', 
                           va='center', 
                           fontsize=fontsize,
                           color=text_color)
    
    ax_saliency.axis('off')
    saliency_path = output_path.replace('.png', '_saliency_only.png')
    plt.savefig(saliency_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Individual visualizations saved to {gradcam_path} and {saliency_path}")

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths"""
    def __getitem__(self, index):
        # Get the image, label as usual
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path

def create_comparative_analysis(gradcam_importances, saliency_importances, motifs, output_dir, dataset_type, sequence_id):
    """
    Create a comparative analysis between GradCAM and Saliency results showing
    biological differences in what each method highlights.
    
    Parameters:
    -----------
    gradcam_importances : list
        List of (dinucleotide, count, importance) tuples from GradCAM
    saliency_importances : list
        List of (dinucleotide, count, importance) tuples from Saliency
    motifs : dict
        Dictionary of biologically significant dinucleotides
    output_dir : str
        Directory to save the output
    dataset_type : str
        Type of dataset (donor or acceptor)
    sequence_id : int
        Sequence ID number
    """
    # Create figure
    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    ax.axis('off')
    
    # Title
    site_type = "Donor (5' Splice Site)" if "_don" in dataset_type else "Acceptor (3' Splice Site)"
    ax.text(0.5, 0.98, f"Biological Significance Analysis: {site_type} - Sequence {sequence_id}", 
            fontsize=16, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
    
    # Extract top 10 dinucleotides from each method
    top_gradcam = gradcam_importances[:10]
    top_saliency = saliency_importances[:10]
    
    # Find common dinucleotides
    gradcam_set = set([d[0] for d in top_gradcam])
    saliency_set = set([d[0] for d in top_saliency])
    common_dinucs = gradcam_set.intersection(saliency_set)
    
    # Calculate biological relevance percentages
    gradcam_bio_relevant = sum(1 for d in top_gradcam if d[0] in motifs)
    saliency_bio_relevant = sum(1 for d in top_saliency if d[0] in motifs)
    
    gradcam_percent = (gradcam_bio_relevant / len(top_gradcam)) * 100 if top_gradcam else 0
    saliency_percent = (saliency_bio_relevant / len(top_saliency)) * 100 if top_saliency else 0
    
    # Add method comparison summary
    ax.text(0.05, 0.92, "Method Comparison:", fontsize=14, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.05, 0.88, f"• GradCAM: {gradcam_bio_relevant}/10 biologically significant dinucleotides ({gradcam_percent:.1f}%)", 
            fontsize=12, ha='left', va='top', transform=ax.transAxes)
    ax.text(0.05, 0.84, f"• Saliency: {saliency_bio_relevant}/10 biologically significant dinucleotides ({saliency_percent:.1f}%)", 
            fontsize=12, ha='left', va='top', transform=ax.transAxes)
    ax.text(0.05, 0.80, f"• Common dinucleotides between methods: {len(common_dinucs)}/10", 
            fontsize=12, ha='left', va='top', transform=ax.transAxes)
    
    # Create table for GradCAM
    ax.text(0.25, 0.75, "GradCAM Top Dinucleotides", fontsize=14, ha='center', va='top', transform=ax.transAxes, fontweight='bold')
    
    y_pos = 0.70
    for i, (dinuc, count, importance) in enumerate(top_gradcam):
        if dinuc in motifs:
            note_text = f"{i+1}. {dinuc}: {importance:.3f} - {motifs[dinuc]}"
            text_color = 'red'
        else:
            note_text = f"{i+1}. {dinuc}: {importance:.3f}"
            text_color = 'black'
        
        ax.text(0.05, y_pos, note_text, fontsize=10, ha='left', va='top', transform=ax.transAxes, color=text_color)
        y_pos -= 0.035
    
    # Create table for Saliency
    ax.text(0.75, 0.75, "Saliency Top Dinucleotides", fontsize=14, ha='center', va='top', transform=ax.transAxes, fontweight='bold')
    
    y_pos = 0.70
    for i, (dinuc, count, importance) in enumerate(top_saliency):
        if dinuc in motifs:
            note_text = f"{i+1}. {dinuc}: {importance:.3f} - {motifs[dinuc]}"
            text_color = 'red'
        else:
            note_text = f"{i+1}. {dinuc}: {importance:.3f}"
            text_color = 'black'
        
        ax.text(0.55, y_pos, note_text, fontsize=10, ha='left', va='top', transform=ax.transAxes, color=text_color)
        y_pos -= 0.035
    
    # Add biological interpretation
    ax.text(0.5, 0.38, "Biological Interpretation", fontsize=14, ha='center', va='top', transform=ax.transAxes, fontweight='bold')
    
    if "_don" in dataset_type:
        ax.text(0.05, 0.33, "Donor Splice Site Interpretation:", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
        
        # Add interpretation for donor sites
        interpretations = [
            "• GT is the canonical dinucleotide at 5' splice sites (exon-intron boundary)",
            "• AG often precedes GT in donor sites (last 2 nucleotides of exon)",
            "• High importance of these dinucleotides indicates model correctly identifies",
            "  the canonical splicing signal"
        ]
        
        y_pos = 0.29
        for interp in interpretations:
            ax.text(0.05, y_pos, interp, fontsize=11, ha='left', va='top', transform=ax.transAxes)
            y_pos -= 0.04
            
    elif "_acc" in dataset_type:
        ax.text(0.05, 0.33, "Acceptor Splice Site Interpretation:", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
        
        # Add interpretation for acceptor sites
        interpretations = [
            "• AG is the canonical dinucleotide at 3' splice sites (intron-exon boundary)",
            "• CT/TC/TT patterns indicate polypyrimidine tract upstream of the AG",
            "• High importance of these dinucleotides indicates model correctly identifies",
            "  the canonical acceptor signal and polypyrimidine tract"
        ]
        
        y_pos = 0.29
        for interp in interpretations:
            ax.text(0.05, y_pos, interp, fontsize=11, ha='left', va='top', transform=ax.transAxes)
            y_pos -= 0.04
    
    # Method comparison
    ax.text(0.05, 0.14, "GradCAM vs. Saliency Comparison:", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    
    # Add method comparison text
    if gradcam_percent > saliency_percent:
        ax.text(0.05, 0.10, "• GradCAM appears more focused on biologically significant dinucleotides", 
                fontsize=11, ha='left', va='top', transform=ax.transAxes)
    elif saliency_percent > gradcam_percent:
        ax.text(0.05, 0.10, "• Saliency appears more focused on biologically significant dinucleotides", 
                fontsize=11, ha='left', va='top', transform=ax.transAxes)
    else:
        ax.text(0.05, 0.10, "• Both methods are equally effective at identifying biologically significant dinucleotides", 
                fontsize=11, ha='left', va='top', transform=ax.transAxes)
    
    # Add note about common dinucleotides
    if common_dinucs:
        common_text = "• Common biologically significant dinucleotides: " + ", ".join([d for d in common_dinucs if d in motifs])
        ax.text(0.05, 0.06, common_text if [d for d in common_dinucs if d in motifs] else "• No common biologically significant dinucleotides", 
                fontsize=11, ha='left', va='top', transform=ax.transAxes)
    
    # Save the analysis
    output_path = os.path.join(output_dir, f"comparative_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparative analysis saved to {output_path}")
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, f"comparative_analysis.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Rank", "Dinucleotide", "Importance", "Biologically Significant", "Biological Role"])
        
        for i, (dinuc, count, importance) in enumerate(top_gradcam):
            writer.writerow([
                "GradCAM", 
                i+1, 
                dinuc, 
                f"{importance:.4f}", 
                "Yes" if dinuc in motifs else "No",
                motifs.get(dinuc, "")
            ])
            
        for i, (dinuc, count, importance) in enumerate(top_saliency):
            writer.writerow([
                "Saliency", 
                i+1, 
                dinuc, 
                f"{importance:.4f}", 
                "Yes" if dinuc in motifs else "No",
                motifs.get(dinuc, "")
            ])
    
    print(f"Comparative analysis CSV saved to {csv_path}")

def process_sequence(seq_number, is_positive, model, target_layer, transform, config, results_df, base_output_dir, device, 
                    quick_mode=False, fragment_analysis=True, section_analysis=True, use_precomputed_maps=False,
                    classification=None):
    """Process a single sequence for visualization with optimization options"""
    # Check if sequence exists in results with the correct positive/negative status
    sequence_with_suffix = f"seq_{seq_number}_{'pos' if is_positive else 'neg'}"
    seq_entry = results_df[results_df['Sequence'] == sequence_with_suffix]
    
    if seq_entry.empty:
        print(f"Warning: Sequence {sequence_with_suffix} not found in results, but proceeding anyway.")
        # Create a dummy entry with default values
        classification = classification or "Unknown"
        probability = 0.0
    else:
        # Get the classification and probability from results if available
        classification = classification or seq_entry['Classification'].values[0]
        probability = seq_entry['Probability'].values[0]
    
    print(f"Selected sequence: {sequence_with_suffix}")
    print(f"Model classification: {classification}, Probability: {probability:.4f}")
    
    # Calculate the actual classification type based on model prediction and user's selection
    # (in case the classification from results_df is incorrect)
    predicted_class = 1 if probability >= 0.5 else 0
    actual_class = 1 if is_positive else 0
    
    if predicted_class == 1 and actual_class == 1:
        correct_classification = "TP"
    elif predicted_class == 0 and actual_class == 0:
        correct_classification = "TN"
    elif predicted_class == 1 and actual_class == 0:
        correct_classification = "FP"
    else:  # predicted_class == 0 and actual_class == 1
        correct_classification = "FN"
    
    # Verify that our calculated classification matches the one from results_df
    if classification != correct_classification:
        print(f"Warning: Classification from results ({classification}) doesn't match calculated classification ({correct_classification}).")
        print(f"Using calculated classification: {correct_classification}")
        classification = correct_classification
    
    # Create directory for this sequence
    sequence_with_suffix = f"seq_{seq_number}_{'pos' if is_positive else 'neg'}"
    seq_output_dir = os.path.join(base_output_dir, sequence_with_suffix)
    os.makedirs(seq_output_dir, exist_ok=True)
    
    # Create cache directory for activation maps
    cache_dir = os.path.join(seq_output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file paths
    gradcam_cache = os.path.join(cache_dir, "gradcam.npy")
    saliency_cache = os.path.join(cache_dir, "saliency.npy")
    
    # Check if we can use precomputed activation maps
    if use_precomputed_maps and os.path.exists(gradcam_cache) and os.path.exists(saliency_cache):
        print("Using precomputed activation maps...")
        gradcam = np.load(gradcam_cache)
        saliency = np.load(saliency_cache)
        
        # For the combined visualization, we need the original image too
        if is_positive:
            image_path = os.path.join(config['test_dir'], "pos", f"seq_{seq_number}.png")
        else:
            image_path = os.path.join(config['test_dir'], "neg", f"seq_{seq_number}.png")
            
        try:
            original_image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Original image file {image_path} not found.")
            original_image = None
    else:
        # Find the image path based on user choice
        if is_positive:
            image_path = os.path.join(config['test_dir'], "pos", f"seq_{seq_number}.png")
        else:
            image_path = os.path.join(config['test_dir'], "neg", f"seq_{seq_number}.png")
        
        print(f"Using image path: {image_path}")
        
        # Try to load the image
        try:
            original_image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file {image_path} not found.")
            return
            
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Check if model was provided (might not be if using precomputed results)
        if model is None:
            print("Error: Model not available. Cannot generate activation maps.")
            return
        
        # Initialize Grad-CAM
        grad_cam = GradCAM(model, target_layer)
        
        # Generate Grad-CAM
        print("Generating Grad-CAM...")
        gradcam = grad_cam.generate_cam(image_tensor)
        
        # Generate saliency map
        print("Generating saliency map...")
        saliency = generate_saliency_map(model, image_tensor.clone())
        
        # Save activation maps to cache
        np.save(gradcam_cache, gradcam)
        np.save(saliency_cache, saliency)
        print("Activation maps saved to cache.")
    
    # Create combined visualization with original image, Grad-CAM, and saliency
    if original_image is not None:
        combined_output_path = os.path.join(seq_output_dir, "combined_visualization.png")
        create_combined_visualization(
            original_image, 
            gradcam, 
            saliency, 
            combined_output_path,
            f"seq_{seq_number}.png (Class: {'pos' if is_positive else 'neg'})",
            classification
        )
    
    # Get global min and max for normalization
    gradcam_min = gradcam.min()
    gradcam_max = gradcam.max()
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    
    print(f"Grad-CAM range: {gradcam_min:.6f} to {gradcam_max:.6f}")
    print(f"Saliency range: {saliency_min:.6f} to {saliency_max:.6f}")
    
    # Get the sequence from files based on classification
    sequence_file = config['sequence_file_pos'] if is_positive else config['sequence_file_neg']
    print(f"\nUsing DNA sequence file: {sequence_file}")
    
    # Print information about the classification and file
    print(f"Classification: {correct_classification} (is_positive={is_positive}, prediction={predicted_class})")
    print(f"Sequence number: {seq_number}")
    
    with open(sequence_file, 'r') as f:
        lines = f.readlines()
        print(f"Total lines in sequence file: {len(lines)}")
        
        # Check if the line exists in the file
        if 0 <= seq_number - 1 < len(lines):
            sequence = lines[seq_number - 1].strip()
            print(f"Retrieved sequence from line {seq_number} (index {seq_number-1})")
        else:
            # The sequence numbers in test sets might be offset (e.g., seq_30001)
            # Try to adjust based on common patterns
            if seq_number >= 30000:  # Test set sequence
                adjusted_index = seq_number - 30000
                print(f"Sequence number {seq_number} appears to be from test set. Trying adjusted index {adjusted_index}")
                if 0 <= adjusted_index - 1 < len(lines):
                    sequence = lines[adjusted_index - 1].strip()
                    print(f"Retrieved sequence from adjusted line {adjusted_index} (index {adjusted_index-1})")
                else:
                    print(f"Error: Adjusted index {adjusted_index} is out of range for file {sequence_file} with {len(lines)} lines")
                    return
            else:
                print(f"Error: Sequence number {seq_number} (line {seq_number}) is out of range for file {sequence_file} with {len(lines)} lines")
                return
    
    print(f"Sequence length: {len(sequence)}")
    
    # Extract dataset type from config for biological context
    dataset_type = os.path.basename(config['test_dir'])
    
    # Continue with the rest of the processing...
    # Print full sequence for verification (only in non-quick mode)
    if not quick_mode:
        print("\nFULL SEQUENCE FOR VERIFICATION:")
        print(sequence)
        print("\n")
    
    # Get splice site motifs based on dataset type
    motifs = get_splice_site_motifs('donor' if '_don' in dataset_type else 'acceptor')
    
    # Create full-sequence dinucleotide table
    full_dinucleotide_grid = create_dinucleotide_table(sequence)
    
    # Identify top focus regions for Grad-CAM and saliency
    focus_regions = identify_top_focus_regions(gradcam, saliency, sequence, full_dinucleotide_grid)
    
    # Print and save focus regions summary
    print_focus_regions_summary(focus_regions, sequence, dataset_type, seq_output_dir)
    
    # Save sequence to a text file for easier reference along with biological context
    with open(os.path.join(seq_output_dir, "sequence_info.txt"), 'w') as f:
        f.write(f"Sequence Number: {seq_number}\n")
        f.write(f"Sequence Identifier: {sequence_with_suffix}\n")
        f.write(f"Sequence Length: {len(sequence)}\n")
        f.write(f"Model Prediction: {'Positive' if predicted_class == 1 else 'Negative'} (Probability: {probability:.4f})\n")
        f.write(f"Actual Class: {'Positive' if is_positive else 'Negative'}\n")
        f.write(f"Classification: {correct_classification}\n\n")
        f.write("FULL SEQUENCE:\n")
        f.write(sequence)
        
        # Add biological context
        f.write("\n\nBIOLOGICAL CONTEXT:\n")
        if '_don' in dataset_type:
            f.write("This sequence represents a donor splice site (5' splice site).\n")
            f.write("Key features to look for:\n")
            f.write("- GT dinucleotide at the exon-intron boundary (canonical)\n")
            f.write("- GC can be a non-canonical donor site (rare)\n")
            f.write("- Donor consensus sequence: MAG|GTRAGT\n")
        elif '_acc' in dataset_type:
            f.write("This sequence represents an acceptor splice site (3' splice site).\n")
            f.write("Key features to look for:\n")
            f.write("- AG dinucleotide at the intron-exon boundary (canonical)\n")
            f.write("- Polypyrimidine tract (CT/TC repeats) upstream of AG\n")
            f.write("- Branch point typically 20-40 nucleotides upstream\n")
            f.write("- Acceptor consensus sequence: TTTCAG|G\n")
    
    # Process sequence fragments if enabled
    if fragment_analysis:
        print("\nExtracting important sequence fragments...")
        important_fragments = extract_important_sequence_fragments(sequence, gradcam, saliency)
        
        if important_fragments:
            print(f"Found {len(important_fragments)} important sequence fragments")
            # Create visualizations and analysis for these fragments
            # In quick mode, generate only summary visualizations
            visualize_sequence_fragments(sequence, important_fragments, gradcam, saliency, 
                                         dataset_type, seq_output_dir, quick_mode=quick_mode)
        else:
            print("No important sequence fragments found.")
    
    # Process grid sections if enabled
    if section_analysis:
        # Divide the sequence into 6×6 sections
        section_size = len(sequence) // 6
        print(f"\nProcessing grid sections (size: {section_size}×{section_size})...")
        
        # Create full-sequence dinucleotide table
        full_dinucleotide_grid = create_dinucleotide_table(sequence)
        
        # Initialize collections for important dinucleotides across all sections
        all_gradcam_importances = []
        all_saliency_importances = []
        
        # In quick mode, only process the sections with highest activation
        if quick_mode:
            # Calculate average activation for each section
            section_activations = []
            for i in range(6):
                for j in range(6):
                    start_row = i * section_size
                    end_row = start_row + section_size if i < 5 else len(sequence)
                    start_col = j * section_size
                    end_col = start_col + section_size if j < 5 else len(sequence)
                    
                    # Get activation for this section
                    gradcam_part = gradcam[start_row:end_row, start_col:end_col]
                    saliency_part = saliency[start_row:end_row, start_col:end_col]
                    
                    avg_activation = (np.mean(gradcam_part) + np.mean(saliency_part)) / 2
                    section_activations.append((i, j, avg_activation))
            
            # Sort sections by activation and take top 3
            section_activations.sort(key=lambda x: x[2], reverse=True)
            sections_to_process = section_activations[:3]
            print(f"Quick mode: Processing only top 3 sections with highest activation")
        else:
            # Process all 36 sections
            sections_to_process = [(i, j, 0) for i in range(6) for j in range(6)]
        
        # Process selected sections
        for idx, (i, j, _) in enumerate(sections_to_process):
            section_idx = i * 6 + j + 1
            print(f"Processing section {section_idx} (grid position {i+1},{j+1})...")
            
            # Calculate start and end indices for this section
            start_row = i * section_size
            end_row = start_row + section_size if i < 5 else len(sequence)
            start_col = j * section_size
            end_col = start_col + section_size if j < 5 else len(sequence)
            
            # Get sequence parts for rows and columns
            seq_row_part = sequence[start_row:end_row]
            seq_col_part = sequence[start_col:end_col]
            
            # Get visualization parts
            gradcam_part = gradcam[start_row:end_row, start_col:end_col]
            saliency_part = saliency[start_row:end_row, start_col:end_col]
            
            # Create dinucleotide grid for this section
            section_dinucleotide_grid = []
            for m in range(len(seq_row_part)):
                row_data = []
                for n in range(len(seq_col_part)):
                    dinuc = seq_row_part[m] + seq_col_part[n]
                    row_data.append(dinuc)
                section_dinucleotide_grid.append(row_data)
            section_dinucleotide_grid = np.array(section_dinucleotide_grid)
            
            # Identify important dinucleotides in this section
            section_gradcam_importances = identify_important_dinucleotides(gradcam_part, section_dinucleotide_grid)
            section_saliency_importances = identify_important_dinucleotides(saliency_part, section_dinucleotide_grid)
            
            # Add to the overall collections
            all_gradcam_importances.extend(section_gradcam_importances)
            all_saliency_importances.extend(section_saliency_importances)
            
            # Create visualization for this section
            output_path = os.path.join(seq_output_dir, f"section_{section_idx}_grid_R{i+1}-C{j+1}.png")
            visualize_section(seq_row_part, seq_col_part, gradcam_part, saliency_part, 
                             gradcam_min, gradcam_max, saliency_min, saliency_max,
                             output_path, section_idx, i+1, j+1,
                             dataset_type=dataset_type, is_positive=is_positive)
            
            # Save section information to text file (skip in quick mode)
            if not quick_mode:
                with open(os.path.join(seq_output_dir, f"section_{section_idx}_info.txt"), 'w') as f:
                    f.write(f"Section {section_idx} (Grid Position R{i+1}-C{j+1})\n")
                    f.write(f"Row indices: {start_row}:{end_row}\n")
                    f.write(f"Column indices: {start_col}:{end_col}\n\n")
                    f.write("ROW SEQUENCE PART:\n")
                    f.write(seq_row_part)
                    f.write("\n\nCOLUMN SEQUENCE PART:\n")
                    f.write(seq_col_part)
                    
                    # Add section-specific analysis
                    f.write("\n\nIMPORTANT DINUCLEOTIDES IN THIS SECTION:\n")
                    f.write("GradCAM top 5:\n")
                    for idx, (dinuc, count, importance) in enumerate(section_gradcam_importances[:5]):
                        bio_note = f" - {motifs[dinuc]}" if dinuc in motifs else ""
                        f.write(f"{idx+1}. {dinuc}: {importance:.4f}{bio_note}\n")
                    
                    f.write("\nSaliency top 5:\n")
                    for idx, (dinuc, count, importance) in enumerate(section_saliency_importances[:5]):
                        bio_note = f" - {motifs[dinuc]}" if dinuc in motifs else ""
                        f.write(f"{idx+1}. {dinuc}: {importance:.4f}{bio_note}\n")
        
        # Process combined dinucleotide importances
        if not quick_mode or len(sections_to_process) > 0:
            print("\nGenerating overall dinucleotide importance analysis...")
            # Combine and consolidate the dinucleotide importances
            gradcam_consolidated = {}
            saliency_consolidated = {}
            
            for dinuc, count, importance in all_gradcam_importances:
                if dinuc not in gradcam_consolidated:
                    gradcam_consolidated[dinuc] = {'count': 0, 'importance_sum': 0.0}
                gradcam_consolidated[dinuc]['count'] += count
                gradcam_consolidated[dinuc]['importance_sum'] += (importance * count)
            
            for dinuc, count, importance in all_saliency_importances:
                if dinuc not in saliency_consolidated:
                    saliency_consolidated[dinuc] = {'count': 0, 'importance_sum': 0.0}
                saliency_consolidated[dinuc]['count'] += count
                saliency_consolidated[dinuc]['importance_sum'] += (importance * count)
            
            # Calculate average importance and prepare final results
            gradcam_final = []
            for dinuc, stats in gradcam_consolidated.items():
                avg_importance = stats['importance_sum'] / stats['count'] if stats['count'] > 0 else 0
                gradcam_final.append((dinuc, stats['count'], avg_importance))
            
            saliency_final = []
            for dinuc, stats in saliency_consolidated.items():
                avg_importance = stats['importance_sum'] / stats['count'] if stats['count'] > 0 else 0
                saliency_final.append((dinuc, stats['count'], avg_importance))
            
            # Sort by average importance
            gradcam_final = sorted(gradcam_final, key=lambda x: x[2], reverse=True)
            saliency_final = sorted(saliency_final, key=lambda x: x[2], reverse=True)
            
            # Save overall dinucleotide importance to CSV
            with open(os.path.join(seq_output_dir, "overall_dinucleotide_importance.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Method", "Dinucleotide", "Count", "Average Importance", "Biologically Significant", "Biological Role"])
                
                for dinuc, count, importance in gradcam_final:
                    writer.writerow([
                        "GradCAM", 
                        dinuc, 
                        count, 
                        f"{importance:.4f}", 
                        "Yes" if dinuc in motifs else "No",
                        motifs.get(dinuc, "")
                    ])
                
                for dinuc, count, importance in saliency_final:
                    writer.writerow([
                        "Saliency", 
                        dinuc, 
                        count, 
                        f"{importance:.4f}", 
                        "Yes" if dinuc in motifs else "No",
                        motifs.get(dinuc, "")
                    ])
            
            # Create comparative analysis between GradCAM and Saliency
            create_comparative_analysis(gradcam_final, saliency_final, motifs, seq_output_dir, dataset_type, seq_number)
            
            # Create a dinucleotide color reference guide (skip in quick mode)
            if not quick_mode:
                create_dinucleotide_reference(seq_output_dir)
    
    print(f"\nAll visualizations for sequence {sequence_with_suffix} have been saved to {seq_output_dir}")
    print("=" * 80)

def extract_important_sequence_fragments(sequence, gradcam, saliency, threshold=0.7):
    """
    Extract DNA sequence fragments from rows with high activation in GradCAM and Saliency maps.
    
    Parameters:
    -----------
    sequence : str
        The DNA sequence
    gradcam : numpy.ndarray
        GradCAM activation map
    saliency : numpy.ndarray
        Saliency activation map
    threshold : float
        Activation threshold for importance
        
    Returns:
    --------
    dict : Dictionary with sequence fragments and their importance scores
    """
    # Normalize GradCAM and Saliency maps
    gradcam_norm = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Find rows with high activation in GradCAM
    gradcam_row_means = np.mean(gradcam_norm, axis=1)  # Calculate mean activation for each row
    high_gradcam_rows = np.where(gradcam_row_means > threshold)[0]
    
    # Find rows with high activation in Saliency
    saliency_row_means = np.mean(saliency_norm, axis=1)
    high_saliency_rows = np.where(saliency_row_means > threshold)[0]
    
    # Combine the high activation rows (union)
    high_activation_rows = np.union1d(high_gradcam_rows, high_saliency_rows)
    
    # Group contiguous rows to form longer sequence fragments
    fragments = []
    if len(high_activation_rows) > 0:
        # Initialize the first fragment
        current_fragment = {
            'start': high_activation_rows[0],
            'end': high_activation_rows[0],
            'gradcam_score': gradcam_row_means[high_activation_rows[0]],
            'saliency_score': saliency_row_means[high_activation_rows[0]]
        }
        
        # Process the rest of the rows
        for i in range(1, len(high_activation_rows)):
            row = high_activation_rows[i]
            prev_row = high_activation_rows[i-1]
            
            # If this row is contiguous with the previous, extend the current fragment
            if row == prev_row + 1:
                current_fragment['end'] = row
                current_fragment['gradcam_score'] = max(current_fragment['gradcam_score'], gradcam_row_means[row])
                current_fragment['saliency_score'] = max(current_fragment['saliency_score'], saliency_row_means[row])
            else:
                # This row starts a new fragment, save the current one and start a new one
                fragments.append(current_fragment)
                current_fragment = {
                    'start': row,
                    'end': row,
                    'gradcam_score': gradcam_row_means[row],
                    'saliency_score': saliency_row_means[row]
                }
        
        # Add the last fragment
        fragments.append(current_fragment)
    
    # Extract the sequence fragments
    for fragment in fragments:
        start = fragment['start']
        end = fragment['end'] + 1  # Add 1 to include the end position
        fragment['sequence'] = sequence[start:end]
        fragment['length'] = end - start
        
        # Calculate the average activations across the entire fragment
        fragment['avg_gradcam'] = np.mean(gradcam_norm[start:end, :])
        fragment['avg_saliency'] = np.mean(saliency_norm[start:end, :])
        fragment['overall_importance'] = (fragment['avg_gradcam'] + fragment['avg_saliency']) / 2
    
    # Sort fragments by overall importance
    fragments.sort(key=lambda x: x['overall_importance'], reverse=True)
    
    return fragments

def analyze_fragment_biology(fragment, dataset_type):
    """
    Analyze the biological significance of a sequence fragment.
    
    Parameters:
    -----------
    fragment : dict
        Dictionary containing sequence fragment information
    dataset_type : str
        Type of dataset (donor or acceptor)
        
    Returns:
    --------
    dict : Dictionary with biological analysis results
    """
    sequence = fragment['sequence']
    analysis = {
        'motifs_found': [],
        'biological_significance': []
    }
    
    # Check for donor site motifs
    if '_don' in dataset_type:
        # Look for GT dinucleotide (canonical donor site)
        gt_positions = [m.start() for m in re.finditer('GT', sequence)]
        if gt_positions:
            analysis['motifs_found'].append(('GT', gt_positions))
            analysis['biological_significance'].append("Contains GT dinucleotide (canonical donor site)")
        
        # Look for GC dinucleotide (non-canonical donor site)
        gc_positions = [m.start() for m in re.finditer('GC', sequence)]
        if gc_positions:
            analysis['motifs_found'].append(('GC', gc_positions))
            analysis['biological_significance'].append("Contains GC dinucleotide (non-canonical donor site)")
            
        # Look for donor consensus pattern (MAG|GTRAGT)
        donor_patterns = [m.span() for m in re.finditer('[AC]AGGT[AG]AGT', sequence)]
        if donor_patterns:
            analysis['motifs_found'].append(('Donor consensus', donor_patterns))
            analysis['biological_significance'].append("Contains canonical donor site consensus sequence")
    
    # Check for acceptor site motifs
    elif '_acc' in dataset_type:
        # Look for AG dinucleotide (canonical acceptor site)
        ag_positions = [m.start() for m in re.finditer('AG', sequence)]
        if ag_positions:
            analysis['motifs_found'].append(('AG', ag_positions))
            analysis['biological_significance'].append("Contains AG dinucleotide (canonical acceptor site)")
        
        # Look for polypyrimidine tract (at least 6 consecutive C/T)
        ppt_patterns = [m.span() for m in re.finditer('[CT]{6,}', sequence)]
        if ppt_patterns:
            analysis['motifs_found'].append(('Polypyrimidine tract', ppt_patterns))
            analysis['biological_significance'].append("Contains polypyrimidine tract (important for acceptor site recognition)")
            
        # Look for branch point consensus (YUNAY)
        branch_patterns = [m.span() for m in re.finditer('[CT][ACGT][CT]A[CT]', sequence)]
        if branch_patterns:
            analysis['motifs_found'].append(('Branch point', branch_patterns))
            analysis['biological_significance'].append("Contains potential branch point consensus sequence")
    
    # General biological features for both types
    # GC content
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) if sequence else 0
    analysis['gc_content'] = gc_content
    
    if gc_content > 0.6:
        analysis['biological_significance'].append("High GC content (may indicate CpG islands or regulatory regions)")
    elif gc_content < 0.4:
        analysis['biological_significance'].append("Low GC content (may indicate AT-rich regions)")
    
    return analysis

def visualize_sequence_fragments(sequence, fragments, gradcam, saliency, dataset_type, output_dir, quick_mode=False):
    """
    Create visualizations for important sequence fragments.
    
    Parameters:
    -----------
    sequence : str
        The full DNA sequence
    fragments : list
        List of important sequence fragments
    gradcam : numpy.ndarray
        GradCAM activation map
    saliency : numpy.ndarray
        Saliency activation map
    dataset_type : str
        Type of dataset (donor or acceptor)
    output_dir : str
        Directory to save the output
    quick_mode : bool
        If True, generate only summary visualizations
    """
    if not fragments:
        print("No important sequence fragments found.")
        return
    
    # Create a summary visualization of all fragments
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Normalize the heatmaps
    gradcam_norm = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Plot the heatmaps
    im1 = ax1.imshow(gradcam_norm, cmap='jet', aspect='auto')
    ax1.set_title("GradCAM Activation with Important Sequence Regions", fontsize=16)
    
    im2 = ax2.imshow(saliency_norm, cmap='hot', aspect='auto')
    ax2.set_title("Saliency Activation with Important Sequence Regions", fontsize=16)
    
    # Add rectangles to highlight important regions
    for i, fragment in enumerate(fragments[:5]):  # Show top 5 fragments
        start = fragment['start']
        end = fragment['end']
        
        # Create a rectangle patch for GradCAM
        rect1 = patches.Rectangle((0, start), gradcam.shape[1], end-start+1, 
                                 linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect1)
        
        # Add a label
        ax1.text(5, start + (end-start)/2, f"Fragment {i+1}", 
                color='white', fontweight='bold', fontsize=10, 
                ha='left', va='center')
        
        # Create a rectangle patch for Saliency
        rect2 = patches.Rectangle((0, start), saliency.shape[1], end-start+1, 
                                 linewidth=2, edgecolor='yellow', facecolor='none')
        ax2.add_patch(rect2)
        
        # Add a label
        ax2.text(5, start + (end-start)/2, f"Fragment {i+1}", 
                color='white', fontweight='bold', fontsize=10, 
                ha='left', va='center')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='GradCAM Activation')
    plt.colorbar(im2, ax=ax2, label='Saliency Activation')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, "important_sequence_fragments_overview.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Important sequence fragments overview saved to {output_path}")
    
    # In quick mode, only create the summary table
    if not quick_mode:
        # Create detailed visualizations for each top fragment
        print(f"Generating detailed visualizations for top {min(5, len(fragments))} fragments...")
        for i, fragment in enumerate(fragments[:5]):  # Process top 5 fragments
            create_fragment_detail_visualization(sequence, fragment, i+1, gradcam_norm, saliency_norm, dataset_type, output_dir)
    
    # Create a summary table
    create_fragment_summary_table(fragments, dataset_type, output_dir)
    
    # Create CSV with sequence fragments for faster processing
    csv_path = os.path.join(output_dir, "sequence_fragments_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Fragment #", "Start", "End", "Length", "Sequence", 
                         "GradCAM Score", "Saliency Score", "Overall Importance"])
        
        for i, fragment in enumerate(fragments):
            writer.writerow([
                i+1,
                fragment['start'],
                fragment['end'],
                fragment['length'],
                fragment['sequence'],
                f"{fragment['avg_gradcam']:.4f}",
                f"{fragment['avg_saliency']:.4f}",
                f"{fragment['overall_importance']:.4f}"
            ])

def create_fragment_detail_visualization(sequence, fragment, fragment_idx, gradcam_norm, saliency_norm, dataset_type, output_dir):
    """
    Create a detailed visualization for a specific sequence fragment.
    
    Parameters:
    -----------
    sequence : str
        The full DNA sequence
    fragment : dict
        Dictionary containing fragment information
    fragment_idx : int
        Index of the fragment (for labeling)
    gradcam_norm : numpy.ndarray
        Normalized GradCAM activation map
    saliency_norm : numpy.ndarray
        Normalized Saliency activation map
    dataset_type : str
        Type of dataset (donor or acceptor)
    output_dir : str
        Directory to save the output
    """
    start = fragment['start']
    end = fragment['end']
    fragment_seq = fragment['sequence']
    
    # Analyze the biological significance of this fragment
    bio_analysis = analyze_fragment_biology(fragment, dataset_type)
    
    # Create a visualization
    fig = plt.figure(figsize=(14, 10))
    
    # Define grid for subplots
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 2])
    
    # Extract row activations
    gradcam_row_activations = np.mean(gradcam_norm[start:end+1, :], axis=1)
    saliency_row_activations = np.mean(saliency_norm[start:end+1, :], axis=1)
    
    # Plot the sequence
    ax1 = plt.subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_title(f"Fragment {fragment_idx}: Positions {start}-{end} (Length: {fragment['length']})", fontsize=16)
    
    # Display the sequence with color-coded nucleotides
    nucleotide_colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    for i, nucleotide in enumerate(fragment_seq):
        color = nucleotide_colors.get(nucleotide, 'black')
        ax1.text(i*0.8/len(fragment_seq), 0.5, nucleotide, 
                transform=ax1.transAxes, fontsize=14, color=color, 
                fontweight='bold', ha='center', va='center')
    
    # Plot GradCAM and Saliency for this region
    ax2 = plt.subplot(gs[1, 0])
    ax2.imshow(gradcam_norm[start:end+1, :], cmap='jet', aspect='auto')
    ax2.set_title(f"GradCAM (Avg: {fragment['avg_gradcam']:.4f})", fontsize=14)
    ax2.set_ylabel("Position in Fragment")
    ax2.set_xlabel("Position in Sequence")
    
    ax3 = plt.subplot(gs[1, 1])
    ax3.imshow(saliency_norm[start:end+1, :], cmap='hot', aspect='auto')
    ax3.set_title(f"Saliency (Avg: {fragment['avg_saliency']:.4f})", fontsize=14)
    ax3.set_ylabel("Position in Fragment")
    ax3.set_xlabel("Position in Sequence")
    
    # Plot biological analysis
    ax4 = plt.subplot(gs[2, :])
    ax4.axis('off')
    
    # Title for biological analysis
    ax4.text(0.5, 0.95, "Biological Significance Analysis", fontsize=14, ha='center', va='top', fontweight='bold')
    
    # Add motifs found
    y_pos = 0.85
    if bio_analysis['motifs_found']:
        ax4.text(0.05, y_pos, "Motifs Found:", fontsize=12, ha='left', va='top', fontweight='bold')
        y_pos -= 0.05
        
        for motif, positions in bio_analysis['motifs_found']:
            if isinstance(positions[0], tuple):  # If positions are spans
                pos_str = ", ".join([f"{start}-{end}" for start, end in positions])
            else:  # If positions are single indices
                pos_str = ", ".join([str(pos) for pos in positions])
                
            ax4.text(0.1, y_pos, f"• {motif}: at positions {pos_str}", fontsize=11, ha='left', va='top')
            y_pos -= 0.05
    
    # Add biological significance
    if bio_analysis['biological_significance']:
        y_pos -= 0.05
        ax4.text(0.05, y_pos, "Biological Significance:", fontsize=12, ha='left', va='top', fontweight='bold')
        y_pos -= 0.05
        
        for significance in bio_analysis['biological_significance']:
            ax4.text(0.1, y_pos, f"• {significance}", fontsize=11, ha='left', va='top')
            y_pos -= 0.05
    
    # Add GC content
    y_pos -= 0.05
    ax4.text(0.05, y_pos, f"GC Content: {bio_analysis['gc_content']:.2f}", fontsize=12, ha='left', va='top')
    
    # Add fragment-specific interpretation based on dataset type
    y_pos -= 0.1
    ax4.text(0.05, y_pos, "Interpretation:", fontsize=12, ha='left', va='top', fontweight='bold')
    y_pos -= 0.05
    
    if '_don' in dataset_type:
        if any('GT' in motif for motif, _ in bio_analysis['motifs_found']):
            ax4.text(0.1, y_pos, "• This fragment contains the GT dinucleotide critical for donor splice site recognition", 
                     fontsize=11, ha='left', va='top')
            y_pos -= 0.05
        if any('Donor consensus' in motif for motif, _ in bio_analysis['motifs_found']):
            ax4.text(0.1, y_pos, "• Contains the full donor site consensus sequence, highly likely to be a functional splice site", 
                     fontsize=11, ha='left', va='top')
            y_pos -= 0.05
    elif '_acc' in dataset_type:
        if any('AG' in motif for motif, _ in bio_analysis['motifs_found']):
            ax4.text(0.1, y_pos, "• This fragment contains the AG dinucleotide critical for acceptor splice site recognition", 
                     fontsize=11, ha='left', va='top')
            y_pos -= 0.05
        if any('Polypyrimidine' in motif for motif, _ in bio_analysis['motifs_found']):
            ax4.text(0.1, y_pos, "• Contains a polypyrimidine tract, important for spliceosome assembly at acceptor sites", 
                     fontsize=11, ha='left', va='top')
            y_pos -= 0.05
        if any('Branch' in motif for motif, _ in bio_analysis['motifs_found']):
            ax4.text(0.1, y_pos, "• Contains a potential branch point sequence, critical for the first step of splicing", 
                     fontsize=11, ha='left', va='top')
            y_pos -= 0.05
    
    # Add model interpretation
    y_pos -= 0.05
    ax4.text(0.1, y_pos, f"• Model activation: High {'GradCAM' if fragment['gradcam_score'] > fragment['saliency_score'] else 'Saliency'} activation suggests this region is important for classification", 
             fontsize=11, ha='left', va='top')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure - use fragment_idx for the filename instead of motif_id
    output_path = os.path.join(output_dir, f"fragment_{fragment_idx}_detail.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fragment {fragment_idx} detail visualization saved to {output_path}")

def create_fragment_summary_table(fragments, dataset_type, output_dir):
    """
    Create a summary table of all important sequence fragments.
    
    Parameters:
    -----------
    fragments : list
        List of important sequence fragments
    dataset_type : str
        Type of dataset (donor or acceptor)
    output_dir : str
        Directory to save the output
    """
    if not fragments:
        return
    
    # Create CSV file with fragment information
    csv_path = os.path.join(output_dir, "sequence_fragments_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Fragment #", "Start", "End", "Length", "Sequence", 
                         "GradCAM Score", "Saliency Score", "Overall Importance", 
                         "GC Content", "Biological Significance"])
        
        for i, fragment in enumerate(fragments):
            # Analyze the fragment
            bio_analysis = analyze_fragment_biology(fragment, dataset_type)
            
            writer.writerow([
                i+1,
                fragment['start'],
                fragment['end'],
                fragment['length'],
                fragment['sequence'],
                f"{fragment['avg_gradcam']:.4f}",
                f"{fragment['avg_saliency']:.4f}",
                f"{fragment['overall_importance']:.4f}",
                f"{bio_analysis['gc_content']:.2f}",
                "; ".join(bio_analysis['biological_significance'])
            ])
    
    print(f"Sequence fragments summary saved to {csv_path}")
    
    # Create a visual summary table
    fig, ax = plt.figure(figsize=(12, min(len(fragments), 10) * 0.5 + 3)), plt.gca()
    ax.axis('off')
    
    # Title
    site_type = "Donor (5' Splice Site)" if "_don" in dataset_type else "Acceptor (3' Splice Site)"
    ax.text(0.5, 0.98, f"Important Sequence Fragments Summary: {site_type}", 
            fontsize=16, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
    
    # Create table header
    ax.text(0.05, 0.92, "#", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.10, 0.92, "Position", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.20, 0.92, "Length", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.30, 0.92, "Sequence", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.60, 0.92, "Importance", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.75, 0.92, "Biological Significance", fontsize=12, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    
    # Add horizontal line
    ax.axhline(y=0.90, xmin=0.05, xmax=0.95, color='black', linewidth=1)
    
    # Add each fragment (limit to top 10)
    y_pos = 0.86
    for i, fragment in enumerate(fragments[:10]):
        # Analyze the fragment
        bio_analysis = analyze_fragment_biology(fragment, dataset_type)
        
        # Determine color based on importance
        if fragment['overall_importance'] > 0.7:
            text_color = 'darkred'
        elif fragment['overall_importance'] > 0.5:
            text_color = 'darkblue'
        else:
            text_color = 'black'
        
        ax.text(0.05, y_pos, f"{i+1}", fontsize=11, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.10, y_pos, f"{fragment['start']}-{fragment['end']}", fontsize=11, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.20, y_pos, f"{fragment['length']}", fontsize=11, ha='left', va='top', transform=ax.transAxes)
        
        # Show sequence (truncate if too long)
        seq_text = fragment['sequence']
        if len(seq_text) > 25:
            seq_text = seq_text[:22] + "..."
            
        ax.text(0.30, y_pos, seq_text, fontsize=11, ha='left', va='top', transform=ax.transAxes, color=text_color)
        ax.text(0.60, y_pos, f"{fragment['overall_importance']:.4f}", fontsize=11, ha='left', va='top', transform=ax.transAxes)
        
        # Show biological significance (truncate if too long)
        if bio_analysis['biological_significance']:
            bio_text = bio_analysis['biological_significance'][0]
            if len(bio_analysis['biological_significance']) > 1:
                bio_text += f" (+ {len(bio_analysis['biological_significance'])-1} more)"
        else:
            bio_text = "No significant motifs found"
            
        ax.text(0.75, y_pos, bio_text, fontsize=11, ha='left', va='top', transform=ax.transAxes)
        
        y_pos -= 0.05
    
    # Save the figure
    output_path = os.path.join(output_dir, "sequence_fragments_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visual sequence fragments summary saved to {output_path}")

def create_dinucleotide_reference(output_dir):
    """Create a reference chart of dinucleotide colors and their biological significance"""
    from dna_image_convert_DFCP import generate_fixed_dinucleotide_colors
    
    # Get the color mapping
    dinucleotide_colors = generate_fixed_dinucleotide_colors()
    
    # Create a figure
    fig, ax = plt.figure(figsize=(10, 12)), plt.gca()
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.98, "Dinucleotide Color Reference and Biological Significance", 
            fontsize=16, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
    
    # Get donor and acceptor motifs
    donor_motifs = get_splice_site_motifs('donor')
    acceptor_motifs = get_splice_site_motifs('acceptor')
    branch_motifs = get_splice_site_motifs('branch_point')
    
    # Create a grid of colored squares with dinucleotide labels
    dinucs = list(dinucleotide_colors.keys())
    cols, rows = 4, 4  # 4x4 grid for 16 dinucleotides
    
    for i, dinuc in enumerate(dinucs):
        row, col = i // cols, i % cols
        
        # Calculate position
        x, y = 0.1 + col * 0.2, 0.85 - row * 0.2
        
        # Get color
        color = [c/255 for c in dinucleotide_colors[dinuc]]
        
        # Draw colored square
        rect = patches.Rectangle((x, y), 0.05, 0.05, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Add dinucleotide label
        ax.text(x + 0.07, y + 0.025, dinuc, fontsize=12, ha='left', va='center')
        
        # Add biological significance if applicable
        notes = []
        if dinuc in donor_motifs:
            notes.append(f"Donor: {donor_motifs[dinuc]}")
        if dinuc in acceptor_motifs:
            notes.append(f"Acceptor: {acceptor_motifs[dinuc]}")
        if dinuc in branch_motifs:
            notes.append(f"Branch: {branch_motifs[dinuc]}")
        
        if notes:
            note_text = "\n".join(notes)
            ax.text(x + 0.07, y - 0.03, note_text, fontsize=8, ha='left', va='top', color='red')
    
    # Add splice site diagrams
    ax.text(0.5, 0.25, "Splice Site Consensus Sequences:", fontsize=14, ha='center', va='center', transform=ax.transAxes)
    
    # Donor site diagram
    ax.text(0.3, 0.2, "Donor (5' splice site):", fontsize=12, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.3, 0.15, "exon | intron", fontsize=10, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.3, 0.1, "MAG | GTRAGT", fontsize=10, ha='center', va='center', transform=ax.transAxes)
    
    # Acceptor site diagram
    ax.text(0.7, 0.2, "Acceptor (3' splice site):", fontsize=12, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.7, 0.15, "intron | exon", fontsize=10, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.7, 0.1, "YYNYAG | G", fontsize=10, ha='center', va='center', transform=ax.transAxes)
    
    # Save the reference guide
    plt.savefig(os.path.join(output_dir, "dinucleotide_reference.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dinucleotide reference chart saved to {output_dir}/dinucleotide_reference.png")

def identify_splicing_motifs(sequence, dataset_type):
    """
    Identify important splicing-related sequence motifs in the DNA sequence.
    
    Parameters:
    -----------
    sequence : str
        The DNA sequence to analyze
    dataset_type : str
        Type of splice site ('donor' or 'acceptor')
    
    Returns:
    --------
    dict : Dictionary of identified motifs with positions and scores
    """
    motifs_found = {}
    
    if '_don' in dataset_type:  # Donor site (5' splice site)
        # Look for canonical donor site pattern: MAG|GTRAGT (M=A/C, R=A/G)
        # Core GT is at position 3-4 after exon-intron boundary
        donor_pattern = r'[AC]AG(GT[AG]AGT)'
        for match in re.finditer(donor_pattern, sequence):
            start, end = match.span(1)
            motifs_found[f"Donor_site_{start}"] = {
                'type': 'donor',
                'sequence': match.group(0),
                'core': match.group(1),
                'position': (start, end),
                'description': "Canonical donor site (5' splice site)"
            }
        
        # Look for non-canonical GC donor sites
        gc_donor_pattern = r'[AC]AG(GC[AG]AGT)'
        for match in re.finditer(gc_donor_pattern, sequence):
            start, end = match.span(1)
            motifs_found[f"NC_Donor_site_{start}"] = {
                'type': 'non_canonical_donor',
                'sequence': match.group(0),
                'core': match.group(1),
                'position': (start, end),
                'description': "Non-canonical GC donor site (rare)"
            }
            
    elif '_acc' in dataset_type:  # Acceptor site (3' splice site)
        # Look for canonical acceptor site pattern: (Y)nNYAG|G
        # We'll look for a simpler pattern: NYAG followed by G
        acceptor_pattern = r'([CT]{8,})([ACGT][CT]AG)G'
        for match in re.finditer(acceptor_pattern, sequence):
            ppt_start, ppt_end = match.span(1)  # Polypyrimidine tract
            acc_start, acc_end = match.span(2)   # Acceptor site
            
            motifs_found[f"Acceptor_site_{acc_start}"] = {
                'type': 'acceptor',
                'sequence': match.group(0),
                'core': match.group(2),
                'position': (acc_start, acc_end),
                'description': "Canonical acceptor site (3' splice site)"
            }
            
            motifs_found[f"Polypyrimidine_tract_{ppt_start}"] = {
                'type': 'polypyrimidine_tract',
                'sequence': match.group(1),
                'position': (ppt_start, ppt_end),
                'description': "Polypyrimidine tract upstream of acceptor site"
            }
        
        # Look for branch points (typically 20-40nt upstream of acceptor site)
        # Consensus is YUNAY where A is the branch point
        branch_pattern = r'([CT][ACGT][CT]A[CT])'
        for match in re.finditer(branch_pattern, sequence):
            start, end = match.span(1)
            
            # Check if this potential branch point is 20-40nt upstream of any acceptor site
            is_valid_branch = False
            for acc_id, acc_info in motifs_found.items():
                if acc_info['type'] == 'acceptor':
                    acc_pos = acc_info['position'][0]
                    distance = acc_pos - end
                    if 15 <= distance <= 45:  # Typical distance range
                        is_valid_branch = True
                        break
            
            if is_valid_branch:
                motifs_found[f"Branch_point_{start}"] = {
                    'type': 'branch_point',
                    'sequence': match.group(1),
                    'position': (start, end),
                    'description': "Potential branch point (YUNAY)",
                    'distance_to_acceptor': distance
                }
    
    return motifs_found

def analyze_motif_importance(motifs, gradcam, saliency, sequence_length):
    """
    Analyze the importance of identified splicing motifs based on GradCAM and Saliency.
    
    Parameters:
    -----------
    motifs : dict
        Dictionary of identified motifs
    gradcam : numpy.ndarray
        The GradCAM heatmap for the sequence
    saliency : numpy.ndarray
        The Saliency map for the sequence
    sequence_length : int
        Length of the DNA sequence
    
    Returns:
    --------
    dict : Dictionary of motifs with importance scores
    """
    motif_importance = {}
    
    for motif_id, motif_info in motifs.items():
        start, end = motif_info['position']
        
        # For sequence motifs, we need to look at the corresponding region in the DFCP image
        # For a motif at positions i to j, we look at the square region from (i,i) to (j,j)
        # as well as surrounding context
        
        # Ensure we don't go out of bounds
        context_size = 5  # Add context around the motif
        start_idx = max(0, start - context_size)
        end_idx = min(sequence_length, end + context_size)
        
        # Extract the region from GradCAM and Saliency maps
        gradcam_region = gradcam[start_idx:end_idx, start_idx:end_idx]
        saliency_region = saliency[start_idx:end_idx, start_idx:end_idx]
        
        # Calculate average activation
        gradcam_activation = np.mean(gradcam_region)
        saliency_activation = np.mean(saliency_region)
        
        # Calculate max activation
        gradcam_max = np.max(gradcam_region)
        saliency_max = np.max(saliency_region)
        
        # Store the results
        motif_importance[motif_id] = {
            **motif_info,  # Include all the original info
            'gradcam_avg': gradcam_activation,
            'gradcam_max': gradcam_max,
            'saliency_avg': saliency_activation,
            'saliency_max': saliency_max,
            'overall_importance': (gradcam_activation + saliency_activation) / 2
        }
    
    return motif_importance

def visualize_splicing_motifs(sequence, motif_importance, gradcam, saliency, output_dir, seq_number, dataset_type):
    """
    Create visualizations highlighting the identified splicing motifs.
    
    Parameters:
    -----------
    sequence : str
        The DNA sequence
    motif_importance : dict
        Dictionary of motifs with importance scores
    gradcam : numpy.ndarray
        GradCAM heatmap
    saliency : numpy.ndarray
        Saliency map
    output_dir : str
        Directory to save the output
    seq_number : int
        Sequence ID number
    dataset_type : str
        Type of dataset (donor or acceptor)
    """
    # Create a figure to visualize the sequence with highlighted motifs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Normalize heatmaps for visualization
    gradcam_norm = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Plot the GradCAM heatmap
    ax1.imshow(gradcam_norm, cmap='jet', aspect='auto')
    ax1.set_title("GradCAM Heatmap with Splicing Motifs", fontsize=16)
    ax1.axis('off')
    
    # Plot the Saliency heatmap
    ax2.imshow(saliency_norm, cmap='hot', aspect='auto')
    ax2.set_title("Saliency Map with Splicing Motifs", fontsize=16)
    ax2.axis('off')
    
    # Add rectangles to highlight motifs
    motif_colors = {
        'donor': 'cyan',
        'non_canonical_donor': 'blue',
        'acceptor': 'red',
        'polypyrimidine_tract': 'green',
        'branch_point': 'magenta'
    }
    
    # Sort motifs by importance
    sorted_motifs = sorted(motif_importance.items(), key=lambda x: x[1]['overall_importance'], reverse=True)
    
    # Add rectangles and annotations for each motif
    for motif_id, motif_info in sorted_motifs:
        start, end = motif_info['position']
        color = motif_colors.get(motif_info['type'], 'yellow')
        
        # Add rectangle to GradCAM plot
        rect1 = patches.Rectangle((start, start), end-start, end-start, 
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect1)
        
        # Add rectangle to Saliency plot
        rect2 = patches.Rectangle((start, start), end-start, end-start, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect2)
        
        # Add label
        ax1.text(start, start-5, f"{motif_info['type']}", color=color, 
                fontweight='bold', fontsize=10, ha='left', va='bottom')
    
    # Add a legend for motif types
    legend_elements = [patches.Patch(facecolor='none', edgecolor=color, label=motif_type)
                       for motif_type, color in motif_colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              ncol=len(motif_colors), frameon=True, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, f"splicing_motifs_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Splicing motifs visualization saved to {output_path}")
    
    # Create a more detailed visualization for each motif
    for motif_id, motif_info in sorted_motifs:
        create_motif_detail_plot(sequence, motif_info, gradcam, saliency, output_dir)
    
    # Create a summary table
    create_motif_summary_table(sorted_motifs, output_dir, seq_number, dataset_type)

def create_motif_detail_plot(sequence, motif_info, gradcam, saliency, output_dir):
    """Create a detailed plot for a specific splicing motif"""
    start, end = motif_info['position']
    motif_type = motif_info['type']
    motif_seq = motif_info['sequence']
    
    # Determine the region to display (motif plus context)
    context_size = 10
    start_idx = max(0, start - context_size)
    end_idx = min(len(sequence), end + context_size)
    
    # Extract the region from the sequence
    region_seq = sequence[start_idx:end_idx]
    
    # Extract the region from heatmaps
    gradcam_region = gradcam[start_idx:end_idx, start_idx:end_idx]
    saliency_region = saliency[start_idx:end_idx, start_idx:end_idx]
    
    # Normalize for visualization
    gradcam_region_norm = (gradcam_region - gradcam_region.min()) / (gradcam_region.max() - gradcam_region.min() + 1e-8)
    saliency_region_norm = (saliency_region - saliency_region.min()) / (saliency_region.max() - saliency_region.min() + 1e-8)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot the sequence region with the motif highlighted
    ax1.axis('off')
    ax1.set_title(f"{motif_type}: {motif_seq}", fontsize=16)
    
    # Display the sequence with the motif highlighted
    for i, nucleotide in enumerate(region_seq):
        pos = i + start_idx
        is_in_motif = start <= pos < end
        color = 'red' if is_in_motif else 'black'
        weight = 'bold' if is_in_motif else 'normal'
        ax1.text(0.1 + i*0.8/len(region_seq), 0.5, nucleotide, 
                transform=ax1.transAxes, fontsize=14, color=color, 
                fontweight=weight, ha='center', va='center')
    
    # Plot the GradCAM region
    ax2.imshow(gradcam_region_norm, cmap='jet')
    ax2.set_title(f"GradCAM (Avg: {motif_info['gradcam_avg']:.4f}, Max: {motif_info['gradcam_max']:.4f})", fontsize=14)
    ax2.axis('off')
    
    # Plot the Saliency region
    ax3.imshow(saliency_region_norm, cmap='hot')
    ax3.set_title(f"Saliency (Avg: {motif_info['saliency_avg']:.4f}, Max: {motif_info['saliency_max']:.4f})", fontsize=14)
    ax3.axis('off')
    
    # Add description
    plt.figtext(0.5, 0.01, motif_info['description'], fontsize=14, ha='center')
    
    # Save the figure - create a safe filename from motif_type and position
    safe_filename = f"{motif_type}_{start}_{end}".replace(":", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f"motif_detail_{safe_filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_motif_summary_table(sorted_motifs, output_dir, seq_number, dataset_type):
    """Create a summary table of identified splicing motifs"""
    # Create figure
    fig, ax = plt.figure(figsize=(12, len(sorted_motifs) * 0.5 + 3)), plt.gca()
    ax.axis('off')
    
    # Title
    site_type = "Donor (5' Splice Site)" if "_don" in dataset_type else "Acceptor (3' Splice Site)"
    ax.text(0.5, 0.98, f"Splicing Motif Analysis: {site_type} - Sequence {seq_number}", 
            fontsize=16, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
    
    # Create table header
    ax.text(0.05, 0.92, "Motif Type", fontsize=14, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.25, 0.92, "Sequence", fontsize=14, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.45, 0.92, "Position", fontsize=14, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.65, 0.92, "GradCAM", fontsize=14, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    ax.text(0.80, 0.92, "Saliency", fontsize=14, ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    
    # Add horizontal line
    ax.axhline(y=0.90, xmin=0.05, xmax=0.95, color='black', linewidth=1)
    
    # Add each motif
    y_pos = 0.86
    for i, (motif_id, motif_info) in enumerate(sorted_motifs):
        start, end = motif_info['position']
        
        # Determine color based on motif type
        if motif_info['type'] in ['donor', 'acceptor']:
            text_color = 'red'
        elif motif_info['type'] in ['branch_point', 'polypyrimidine_tract']:
            text_color = 'blue'
        else:
            text_color = 'black'
        
        ax.text(0.05, y_pos, motif_info['type'].replace('_', ' ').title(), 
                fontsize=12, ha='left', va='top', transform=ax.transAxes, color=text_color)
        ax.text(0.25, y_pos, motif_info['sequence'], 
                fontsize=12, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.45, y_pos, f"{start}-{end}", 
                fontsize=12, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.65, y_pos, f"{motif_info['gradcam_avg']:.4f}", 
                fontsize=12, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.80, y_pos, f"{motif_info['saliency_avg']:.4f}", 
                fontsize=12, ha='left', va='top', transform=ax.transAxes)
        
        y_pos -= 0.05
    
    # Add a divider
    ax.axhline(y=y_pos+0.02, xmin=0.05, xmax=0.95, color='black', linewidth=1)
    
    # Add biological interpretation
    y_pos -= 0.05
    ax.text(0.5, y_pos, "Biological Interpretation", 
           fontsize=14, ha='center', va='top', transform=ax.transAxes, fontweight='bold')
    
    y_pos -= 0.07
    if "_don" in dataset_type:
        interpretation = [
            "• The canonical donor site (GT) marks the exon-intron boundary.",
            "• Higher activation around the GT dinucleotide indicates the model recognizes this critical signal.",
            "• The consensus sequence MAG|GTRAGT (where | is the exon-intron boundary) is a strong indicator of splice sites."
        ]
    else:  # acceptor
        interpretation = [
            "• The canonical acceptor site (AG) marks the intron-exon boundary.",
            "• The polypyrimidine tract (series of C/T) upstream of the AG is a key recognition element.",
            "• Branch points (containing the conserved A) are critical for the first step of splicing.",
            "• Higher activation in these regions suggests the model has learned the biological splicing signal."
        ]
    
    for line in interpretation:
        ax.text(0.05, y_pos, line, fontsize=12, ha='left', va='top', transform=ax.transAxes)
        y_pos -= 0.05
    
    # Save the figure
    output_path = os.path.join(output_dir, f"splicing_motifs_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, f"splicing_motifs_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Motif Type", "Sequence", "Position", "Description", 
                         "GradCAM Avg", "GradCAM Max", "Saliency Avg", "Saliency Max", "Overall Importance"])
        
        for motif_id, motif_info in sorted_motifs:
            start, end = motif_info['position']
            writer.writerow([
                motif_info['type'],
                motif_info['sequence'],
                f"{start}-{end}",
                motif_info['description'],
                f"{motif_info['gradcam_avg']:.4f}",
                f"{motif_info['gradcam_max']:.4f}",
                f"{motif_info['saliency_avg']:.4f}",
                f"{motif_info['saliency_max']:.4f}",
                f"{motif_info['overall_importance']:.4f}"
            ])
    
    print(f"Splicing motifs summary saved to {output_path} and {csv_path}")

def create_combined_visualization(original_image, gradcam, saliency, output_path, sequence_id, classification_type):
    """
    Create a combined visualization showing the original image alongside Grad-CAM and saliency maps.
    
    Parameters:
    -----------
    original_image : PIL.Image or torch.Tensor
        The original FCGR image
    gradcam : numpy.ndarray
        The Grad-CAM heatmap
    saliency : numpy.ndarray
        The saliency map
    output_path : str
        Path to save the visualization
    sequence_id : str
        Identifier for the sequence
    classification_type : str
        Classification type (TP, TN, FP, FN)
    """
    # If original_image is a tensor, convert to numpy
    if isinstance(original_image, torch.Tensor):
        # Convert from CxHxW to HxWxC and denormalize
        img_np = original_image.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # Denormalize and clip to [0,1]
    else:
        # Convert PIL image to numpy array
        img_np = np.array(original_image) / 255.0
    
    # Create figure with two rows: original images and overlays
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original image and separate heatmaps
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(sequence_id, fontsize=16)
    axes[0, 0].axis('off')
    
    # Normalize heatmaps
    gradcam_norm = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Display Grad-CAM
    axes[0, 1].imshow(gradcam_norm, cmap='jet')
    axes[0, 1].set_title("Grad-CAM Heatmap", fontsize=16)
    axes[0, 1].axis('off')
    
    # Display Saliency
    axes[0, 2].imshow(saliency_norm, cmap='hot')
    axes[0, 2].set_title("Saliency Map", fontsize=16)
    axes[0, 2].axis('off')
    
    # Row 2: Overlay heatmaps on original image
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title(f"Original (Class: {classification_type})", fontsize=16)
    axes[1, 0].axis('off')
    
    # Overlay Grad-CAM on original
    axes[1, 1].imshow(img_np)
    im_gradcam = axes[1, 1].imshow(gradcam_norm, cmap='jet', alpha=0.7)
    axes[1, 1].set_title("Original + Grad-CAM", fontsize=16)
    axes[1, 1].axis('off')
    plt.colorbar(im_gradcam, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Overlay Saliency on original
    axes[1, 2].imshow(img_np)
    im_saliency = axes[1, 2].imshow(saliency_norm, cmap='hot', alpha=0.7)
    axes[1, 2].set_title("Original + Saliency", fontsize=16)
    axes[1, 2].axis('off')
    plt.colorbar(im_saliency, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved to {output_path}")
    
    # Also create separate files for Grad-CAM and Saliency visualizations
    # Grad-CAM visualization
    fig_gradcam, axes_gradcam = plt.subplots(1, 2, figsize=(12, 6))
    axes_gradcam[0].imshow(img_np)
    axes_gradcam[0].set_title(sequence_id, fontsize=16)
    axes_gradcam[0].axis('off')
    
    axes_gradcam[1].imshow(img_np)
    im_gradcam = axes_gradcam[1].imshow(gradcam_norm, cmap='jet', alpha=0.7)
    axes_gradcam[1].set_title("Original + Grad-CAM", fontsize=16)
    axes_gradcam[1].axis('off')
    plt.colorbar(im_gradcam, ax=axes_gradcam[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    gradcam_path = output_path.replace('.png', '_gradcam.png')
    plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Saliency visualization
    fig_saliency, axes_saliency = plt.subplots(1, 2, figsize=(12, 6))
    axes_saliency[0].imshow(img_np)
    axes_saliency[0].set_title(sequence_id, fontsize=16)
    axes_saliency[0].axis('off')
    
    axes_saliency[1].imshow(img_np)
    im_saliency = axes_saliency[1].imshow(saliency_norm, cmap='hot', alpha=0.7)
    axes_saliency[1].set_title("Original + Saliency", fontsize=16)
    axes_saliency[1].axis('off')
    plt.colorbar(im_saliency, ax=axes_saliency[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    saliency_path = output_path.replace('.png', '_saliency.png')
    plt.savefig(saliency_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Separate visualizations saved to {gradcam_path} and {saliency_path}")

def identify_top_focus_regions(gradcam, saliency, sequence, dinucleotide_grid, top_n=10):
    """
    Identify the top regions where Grad-CAM and saliency focus the most and map to DNA sequence.
    
    Parameters:
    -----------
    gradcam : numpy.ndarray
        The Grad-CAM heatmap
    saliency : numpy.ndarray
        The saliency map
    sequence : str
        The DNA sequence
    dinucleotide_grid : numpy.ndarray
        Grid of dinucleotides
    top_n : int
        Number of top regions to identify
        
    Returns:
    --------
    dict : Dictionary with top regions for Grad-CAM and saliency
    """
    # Normalize heatmaps
    gradcam_norm = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Find top regions for Grad-CAM
    gradcam_top_indices = np.unravel_index(np.argsort(gradcam_norm.ravel())[-top_n:], gradcam_norm.shape)
    gradcam_top_values = gradcam_norm[gradcam_top_indices]
    
    # Find top regions for saliency
    saliency_top_indices = np.unravel_index(np.argsort(saliency_norm.ravel())[-top_n:], saliency_norm.shape)
    saliency_top_values = saliency_norm[saliency_top_indices]
    
    # Map indices to dinucleotides and sequence positions
    gradcam_regions = []
    for i in range(top_n):
        row, col = gradcam_top_indices[0][-(i+1)], gradcam_top_indices[1][-(i+1)]
        value = gradcam_top_values[-(i+1)]
        dinuc = dinucleotide_grid[row, col] if row < len(dinucleotide_grid) and col < len(dinucleotide_grid[0]) else "N/A"
        
        # Map to sequence position (approximate since FCGR isn't a direct 1:1 mapping)
        # Here we identify which part of the sequence this position corresponds to
        seq_pos_row = row
        seq_pos_col = col
        
        gradcam_regions.append({
            'position': (row, col),
            'value': value,
            'dinucleotide': dinuc,
            'seq_position_row': seq_pos_row,
            'seq_position_col': seq_pos_col
        })
    
    saliency_regions = []
    for i in range(top_n):
        row, col = saliency_top_indices[0][-(i+1)], saliency_top_indices[1][-(i+1)]
        value = saliency_top_values[-(i+1)]
        dinuc = dinucleotide_grid[row, col] if row < len(dinucleotide_grid) and col < len(dinucleotide_grid[0]) else "N/A"
        
        # Map to sequence position
        seq_pos_row = row
        seq_pos_col = col
        
        saliency_regions.append({
            'position': (row, col),
            'value': value,
            'dinucleotide': dinuc,
            'seq_position_row': seq_pos_row,
            'seq_position_col': seq_pos_col
        })
    
    return {
        'gradcam': gradcam_regions,
        'saliency': saliency_regions
    }

def print_focus_regions_summary(focus_regions, sequence, dataset_type, output_dir):
    """
    Print and save a summary of the most important regions in the DNA sequence.
    
    Parameters:
    -----------
    focus_regions : dict
        Dictionary with top regions for Grad-CAM and saliency
    sequence : str
        The DNA sequence
    dataset_type : str
        Type of dataset (donor or acceptor)
    output_dir : str
        Directory to save the output
    """
    # Get splice site motifs based on dataset type
    motifs = get_splice_site_motifs('donor' if '_don' in dataset_type else 'acceptor')
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "focus_regions_summary.txt")
    
    with open(summary_path, 'w') as f:
        # Print Grad-CAM focus regions
        f.write("TOP GRAD-CAM FOCUS REGIONS:\n")
        f.write("==========================\n\n")
        
        for i, region in enumerate(focus_regions['gradcam']):
            f.write(f"Region {i+1}:\n")
            f.write(f"  Position in grid: ({region['position'][0]}, {region['position'][1]})\n")
            f.write(f"  Activation value: {region['value']:.4f}\n")
            f.write(f"  Dinucleotide: {region['dinucleotide']}\n")
            
            # Check if this dinucleotide has biological significance
            if region['dinucleotide'] in motifs:
                f.write(f"  Biological significance: {motifs[region['dinucleotide']]}\n")
            else:
                f.write("  Biological significance: None identified\n")
            
            # Show sequence context
            row_pos = region['seq_position_row']
            col_pos = region['seq_position_col']
            context_size = 5
            
            start_row = max(0, row_pos - context_size)
            end_row = min(len(sequence), row_pos + context_size + 1)
            
            f.write(f"  Sequence context (position {row_pos}):\n")
            f.write(f"    {sequence[start_row:end_row]}\n")
            f.write(f"    {' ' * (row_pos - start_row)}^\n\n")
        
        f.write("\n\n")
        
        # Print Saliency focus regions
        f.write("TOP SALIENCY FOCUS REGIONS:\n")
        f.write("==========================\n\n")
        
        for i, region in enumerate(focus_regions['saliency']):
            f.write(f"Region {i+1}:\n")
            f.write(f"  Position in grid: ({region['position'][0]}, {region['position'][1]})\n")
            f.write(f"  Activation value: {region['value']:.4f}\n")
            f.write(f"  Dinucleotide: {region['dinucleotide']}\n")
            
            # Check if this dinucleotide has biological significance
            if region['dinucleotide'] in motifs:
                f.write(f"  Biological significance: {motifs[region['dinucleotide']]}\n")
            else:
                f.write("  Biological significance: None identified\n")
            
            # Show sequence context
            row_pos = region['seq_position_row']
            col_pos = region['seq_position_col']
            context_size = 5
            
            start_row = max(0, row_pos - context_size)
            end_row = min(len(sequence), row_pos + context_size + 1)
            
            f.write(f"  Sequence context (position {row_pos}):\n")
            f.write(f"    {sequence[start_row:end_row]}\n")
            f.write(f"    {' ' * (row_pos - start_row)}^\n\n")
    
    print(f"Focus regions summary saved to {summary_path}")
    
    # Also create a CSV version for easier analysis
    csv_path = os.path.join(output_dir, "focus_regions_summary.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Region", "Row", "Column", "Activation", "Dinucleotide", "Biologically Significant", "Biological Role"])
        
        for i, region in enumerate(focus_regions['gradcam']):
            writer.writerow([
                "GradCAM",
                i+1,
                region['position'][0],
                region['position'][1],
                f"{region['value']:.4f}",
                region['dinucleotide'],
                "Yes" if region['dinucleotide'] in motifs else "No",
                motifs.get(region['dinucleotide'], "")
            ])
        
        for i, region in enumerate(focus_regions['saliency']):
            writer.writerow([
                "Saliency",
                i+1,
                region['position'][0],
                region['position'][1],
                f"{region['value']:.4f}",
                region['dinucleotide'],
                "Yes" if region['dinucleotide'] in motifs else "No",
                motifs.get(region['dinucleotide'], "")
            ])
    
    print(f"Focus regions CSV saved to {csv_path}")
    
    # Print a summary to console
    print("\n===== FOCUS REGIONS SUMMARY =====")
    print("\nTop 5 Grad-CAM focus regions:")
    for i, region in enumerate(focus_regions['gradcam'][:5]):
        bio_sig = f" - {motifs[region['dinucleotide']]}" if region['dinucleotide'] in motifs else ""
        print(f"  {i+1}. Position ({region['position'][0]}, {region['position'][1]}): {region['dinucleotide']} (Activation: {region['value']:.4f}){bio_sig}")
    
    print("\nTop 5 Saliency focus regions:")
    for i, region in enumerate(focus_regions['saliency'][:5]):
        bio_sig = f" - {motifs[region['dinucleotide']]}" if region['dinucleotide'] in motifs else ""
        print(f"  {i+1}. Position ({region['position'][0]}, {region['position'][1]}): {region['dinucleotide']} (Activation: {region['value']:.4f}){bio_sig}")

def main():
    # Model configurations for different datasets
    model_configs = {
        'arab_don': {
            'model_path': 'dna_image_saved_models/resnet50_fixed_arab_don_20250415_234130.pth',
            'model_type': 'resnet50',
            'test_dir': 'Test_Image_fixed/arab_don',
            'sequence_file_pos': 'DRANet/arabidopsis_donor_positive.txt',
            'sequence_file_neg': 'DRANet/arabidopsis_donor_negative.txt'
        },
        'arab_acc': {
            'model_path': 'dna_image_saved_models/resnet50_fixed_arab_acc_20250415_184840.pth',
            'model_type': 'resnet50',
            'test_dir': 'Test_Image_fixed/arab_acc',
            'sequence_file_pos': 'DRANet/arabidopsis_acceptor_positive.txt',
            'sequence_file_neg': 'DRANet/arabidopsis_acceptor_negative.txt'
        },
        'homo_don': {
            'model_path': 'dna_image_saved_models/resnet50_fixed_homo_don_20250416_151039.pth',
            'model_type': 'resnet50',
            'test_dir': 'Test_Image_fixed/homo_don',
            'sequence_file_pos': 'DRANet/homo_donor_positive.txt',
            'sequence_file_neg': 'DRANet/homo_donor_negative.txt'
        },
        'homo_acc': {
            'model_path': 'dna_image_saved_models/resnet50_fixed_homo_acc_20250416_043420.pth', 
            'model_type': 'resnet50',
            'test_dir': 'Test_Image_fixed/homo_acc',
            'sequence_file_pos': 'DRANet/homo_acceptor_positive.txt',
            'sequence_file_neg': 'DRANet/homo_acceptor_negative.txt'
        }
    }
    
    # List available data types
    print("Available data types:")
    for i, data_type in enumerate(model_configs.keys()):
        print(f"{i+1}. {data_type}")
    
    # Ask user to select a data type
    data_type_idx = int(input("Select a data type (1-4): ")) - 1
    data_type = list(model_configs.keys())[data_type_idx]
    config = model_configs[data_type]
    
    print(f"Selected data type: {data_type}")
    print(f"Using model: {config['model_path']}")
    
    # Add options for quick mode
    print("\nOptions:")
    quick_mode = input("Run in quick mode? (fewer visualizations, faster) [y/N]: ").lower() == 'y'
    fragment_analysis = input("Include sequence fragment analysis? [Y/n]: ").lower() != 'n'
    section_analysis = input("Include grid section analysis? [Y/n]: ").lower() != 'n'
    use_precomputed = input("Use precomputed results if available? (much faster) [Y/n]: ").lower() != 'n'
    
    print(f"\nRunning with options: quick_mode={quick_mode}, fragment_analysis={fragment_analysis}, section_analysis={section_analysis}, use_precomputed={use_precomputed}")
    
    # Create output directories - changed to use final_figures/DFCP
    base_output_dir = f"final_figures/DFCP/{data_type}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Check for precomputed results
    results_csv_path = os.path.join(base_output_dir, f"{data_type}_results.csv")
    if use_precomputed and os.path.exists(results_csv_path):
        print(f"Using precomputed results from {results_csv_path}")
        results_df = pd.read_csv(results_csv_path)
        model = None
        target_layer = None
    else:
        # Data transforms - same as used for training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Check if test directory exists
        if not os.path.exists(config['test_dir']):
            print(f"Error: Test directory {config['test_dir']} does not exist.")
            return
        
        # Load test dataset with file paths
        print("Loading test dataset...")
        test_dataset = ImageFolderWithPaths(config['test_dir'], transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Load model
        print("Loading model...")
        model, target_layer = load_model(config['model_path'], config['model_type'])
        
        # Evaluate model on test set
        print("Evaluating model on test set...")
        predictions, labels, probs, images, paths = evaluate_model(model, test_loader, device)
        
        # Create CSV to store results
        results_data = []
        
        for i, (pred, label, prob, path) in enumerate(zip(predictions, labels, probs, paths)):
            # Extract sequence number from path
            seq_number = int(os.path.basename(path).split('_')[1].split('.')[0])
            
            # Determine if it's from pos or neg directory
            is_positive = "pos" in os.path.dirname(path)
            
            # Create a unique identifier that includes both sequence number and pos/neg status
            sequence_id = f"seq_{seq_number}_{'pos' if is_positive else 'neg'}"
            
            # Determine the classification type
            predicted_class = 1 if prob >= 0.5 else 0
            actual_class = 1 if is_positive else 0
            
            if predicted_class == 1 and actual_class == 1:
                class_type = "TP"
            elif predicted_class == 0 and actual_class == 0:
                class_type = "TN"
            elif predicted_class == 1 and actual_class == 0:
                class_type = "FP"
            else:  # predicted_class == 0 and actual_class == 1
                class_type = "FN"
            
            results_data.append({
                'Sequence': f"seq_{seq_number}_{'pos' if is_positive else 'neg'}",  # Add pos/neg suffix directly to Sequence
                'Sequence_ID': sequence_id,
                'Classification': class_type,
                'Probability': float(prob)
            })
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to {results_csv_path}")
        
        # Save summary statistics
        summary_csv_path = os.path.join("final_figures/DFCP", f"{data_type}_summary.csv")
        classification_counts = results_df['Classification'].value_counts()
        summary_df = pd.DataFrame({
            'Classification': classification_counts.index,
            'Count': classification_counts.values
        })
        summary_df.to_csv(summary_csv_path, index=False)
        
        # Print classification summary
        print("\nClassification Summary:")
        for cls_type, count in classification_counts.items():
            print(f"{cls_type}: {count}")
    
    # Show classification counts for this dataset
    print("\nAvailable classifications in this dataset:")
    classification_counts = results_df['Classification'].value_counts()
    for cls_type, count in classification_counts.items():
        print(f"{cls_type}: {count}")
    
    # Transform that will be used for individual images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Ask what type of classification the user wants to analyze
    print("\nWhat type of classification do you want to analyze?")
    print("1. TP (True Positive)")
    print("2. TN (True Negative)")
    print("3. FP (False Positive)")
    print("4. FN (False Negative)")
    print("5. Any (don't filter by classification)")
    
    classification_choice = int(input("Enter your choice (1-5): "))
    target_classification = {
        1: "TP",
        2: "TN",
        3: "FP",
        4: "FN",
        5: None
    }.get(classification_choice)
    
    if target_classification:
        print(f"\nSelected classification: {target_classification}")
        # Filter results to show only sequences of the selected classification
        filtered_df = results_df[results_df['Classification'] == target_classification]
        print(f"Found {len(filtered_df)} sequences with classification {target_classification}")
        
        if len(filtered_df) > 0:
            print("\nAvailable sequences:")
            for idx, row in filtered_df.head(20).iterrows():
                # Parse sequence string to determine if positive or negative
                is_positive = "_pos" in row['Sequence']
                print(f"  {row['Sequence']} (Probability: {row['Probability']:.4f})")
            if len(filtered_df) > 20:
                print(f"  ... and {len(filtered_df) - 20} more")
    else:
        filtered_df = results_df
        print("\nNot filtering by classification type")
    
    # Ask user for number of sequences to visualize and their indices
    print("\nHow many sequences would you like to visualize?")
    num_sequences = int(input("Enter number: "))
    
    # Collect all sequence numbers and types upfront
    sequence_info = []
    
    print("\nEnter the information for all sequences:")
    for i in range(num_sequences):
        while True:
            # Ask for sequence number
            print(f"\nSequence {i+1} of {num_sequences}:")
            seq_input = input("Enter sequence number (seq_): ")
            seq_number = int(seq_input)
            
            # Ask if this is a positive or negative example
            pos_neg_choice = input("Is this a positive (p) or negative (n) example? [p/n]: ").lower()
            is_positive = pos_neg_choice == 'p'
            
            # Create sequence identifier with pos/neg suffix
            sequence_with_suffix = f"seq_{seq_number}_{'pos' if is_positive else 'neg'}"
            
            # Find this sequence in the results
            seq_entry = results_df[results_df['Sequence'] == sequence_with_suffix]
            
            if seq_entry.empty:
                print(f"Warning: Sequence {sequence_with_suffix} not found in results.")
                if input("Do you want to try another sequence? [Y/n]: ").lower() != 'n':
                    continue
                else:
                    # Create dummy values for a sequence not in results
                    classification = "Unknown"
                    probability = 0.5
            else:
                # Get the classification and probability
                classification = seq_entry['Classification'].values[0]
                probability = seq_entry['Probability'].values[0]
                print(f"This is a {classification} sequence (Probability: {probability:.4f})")
            
            # Check if this matches the target classification (if specified)
            if target_classification and classification != target_classification:
                print(f"Warning: This sequence is classified as {classification}, not {target_classification}.")
                if input("Do you want to try another sequence? [Y/n]: ").lower() != 'n':
                    continue
            
            # Ask if user wants to confirm this selection
            if input(f"Confirm selection of seq_{seq_number}_{'pos' if is_positive else 'neg'} ({'Positive' if is_positive else 'Negative'}, {classification})? [Y/n]: ").lower() != 'n':
                break
        
        # Ask if we should use precomputed activation maps if available
        use_precomputed_maps = False
        if use_precomputed:
            sequence_with_suffix = f"seq_{seq_number}_{'pos' if is_positive else 'neg'}"
            cache_dir = os.path.join(base_output_dir, sequence_with_suffix, "cache")
            if os.path.exists(cache_dir):
                use_precomputed_maps = input(f"Found precomputed activation maps for {sequence_with_suffix}. Use them? [Y/n]: ").lower() != 'n'
        
        sequence_info.append({
            'seq_number': seq_number,
            'is_positive': is_positive,
            'classification': classification,
            'probability': probability,
            'use_precomputed_maps': use_precomputed_maps
        })
    
    # Process all sequences with the collected information
    print("\nProcessing all sequences. Please wait...")
    
    for idx, seq_data in enumerate(sequence_info):
        print(f"\n=== Processing sequence {idx+1} of {num_sequences} ===")
        print(f"Sequence: seq_{seq_data['seq_number']}_{'pos' if seq_data['is_positive'] else 'neg'}, Type: {'Positive' if seq_data['is_positive'] else 'Negative'}, Classification: {seq_data['classification']}")
        
        # Process the sequence with optimization options
        process_sequence(
            seq_number=seq_data['seq_number'],
            is_positive=seq_data['is_positive'],
            model=model,
            target_layer=target_layer,
            transform=transform,
            config=config,
            results_df=results_df,
            base_output_dir=base_output_dir,
            device=device,
            quick_mode=quick_mode,
            fragment_analysis=fragment_analysis,
            section_analysis=section_analysis,
            use_precomputed_maps=seq_data.get('use_precomputed_maps', False),
            classification=seq_data.get('classification', None)
        )
    
    # Create a combined reference guide in the main DFCP directory
    if not quick_mode:
        create_dinucleotide_reference("final_figures/DFCP")
    
    print("\nAll requested sequences have been processed successfully!")
    print(f"Results saved to final_figures/DFCP/{data_type}")

if __name__ == "__main__":
    # Create the main output directory
    os.makedirs("final_figures/DFCP", exist_ok=True)
    main() 