import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import random
from tqdm import tqdm
import re

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

# --------------------------
# Set device
# --------------------------
# device = torch.device("cpu")  # Using CPU to avoid memory issues
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# --------------------------
# Define Grad-CAM class
# --------------------------
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
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    target_layer = model.layer4[-1]
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, target_layer

def evaluate_model(model, test_loader, dataset, device):
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []
    all_indices = []  # Track file indices
    all_filenames = []  # Track actual filenames
    all_classes = []  # Track class folders
    
    # Get list of image filenames from dataset
    image_paths = [path for path, _ in dataset.imgs]
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()  # Use sigmoid for binary classification
            preds = (probs > 0.5).float()  # Threshold at 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.cpu())
            
            # Store original indices to identify image files later
            batch_start_idx = batch_idx * test_loader.batch_size
            batch_indices = list(range(batch_start_idx, batch_start_idx + len(images)))
            all_indices.extend(batch_indices)
            
            # Get filenames and class folders for this batch
            batch_filenames = []
            batch_classes = []
            for i in batch_indices:
                if i < len(image_paths):
                    batch_filenames.append(image_paths[i])
                    # Extract class folder (0 or 1)
                    class_folder = os.path.basename(os.path.dirname(image_paths[i]))
                    batch_classes.append(class_folder)
                    
            all_filenames.extend(batch_filenames)
            all_classes.extend(batch_classes)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images, all_indices, all_filenames, all_classes

def get_tp_tn_samples(predictions, labels, images, indices, filenames, classes, num_samples=3):
    # Find True Positives and True Negatives
    tp_indices = np.where((predictions == 1) & (labels == 1))[0]
    tn_indices = np.where((predictions == 0) & (labels == 0))[0]
    
    # Sort indices (to get the first n samples instead of random)
    tp_indices = np.sort(tp_indices)
    tn_indices = np.sort(tn_indices)
    
    # Select samples (take the first num_samples)
    tp_samples = []
    tn_samples = []
    
    if len(tp_indices) > 0:
        selected_tp = tp_indices[:min(num_samples, len(tp_indices))]
        tp_samples = [(images[i], labels[i], indices[i], filenames[i], classes[i]) for i in selected_tp]
    
    if len(tn_indices) > 0:
        selected_tn = tn_indices[:min(num_samples, len(tn_indices))]
        tn_samples = [(images[i], labels[i], indices[i], filenames[i], classes[i]) for i in selected_tn]
    
    return {
        'tp': tp_samples,
        'tn': tn_samples,
        'tp_count': len(tp_indices),
        'tn_count': len(tn_indices)
    }

def extract_seq_number(filename):
    # Extract seq_XXXXX.png from the path
    match = re.search(r'seq_(\d+)\.png', filename)
    if match:
        return f"seq_{match.group(1)}.png"
    return os.path.basename(filename)

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image with specified transparency"""
    # Convert heatmap to RGB
    heatmap_rgb = plt.cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Blend the images
    blended = (1 - alpha) * img + alpha * heatmap_rgb
    
    # Ensure values are in valid range
    blended = np.clip(blended, 0, 1)
    
    return blended

def create_visualization_figure(images_data, title_suffix, output_path):
    # Unpack the data for this figure
    orig_images, saliency_maps, gradcam_heatmaps, image_filenames, image_classes, sample_types = images_data
    
    num_samples = len(orig_images)
    if num_samples == 0:
        return
    
    # Create figure
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    # Handle case where only one sample is available
    if num_samples == 1:
        axs = np.array([axs])
    
    # Increase font size for all text
    plt.rcParams.update({'font.size': 14})  # Increase default font size
    
    for i in range(num_samples):
        # Original Image
        axs[i, 0].imshow(orig_images[i])
        axs[i, 0].set_title(f"{image_filenames[i]} (Class: {image_classes[i]})", fontsize=16)
        axs[i, 0].axis("off")
        
        # Saliency Map - Show raw saliency map for better visibility of focused areas
        im_sal = axs[i, 1].imshow(saliency_maps[i], cmap="hot")
        axs[i, 1].set_title("Saliency Map", fontsize=16)
        axs[i, 1].axis("off")
        
        # Grad-CAM Heatmap with Original Image Overlay
        # Create overlay of original image and Grad-CAM heatmap
        gradcam_overlay = overlay_heatmap(orig_images[i], gradcam_heatmaps[i])
        axs[i, 2].imshow(gradcam_overlay)
        axs[i, 2].set_title("Grad-CAM Heatmap", fontsize=16)
        axs[i, 2].axis("off")
    
    # No suptitle as requested
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualizations saved to {output_path}")

def generate_visualizations(model, target_layer, test_loader, dataset, output_path_base, dataset_name, num_samples=3):
    # Evaluate model on test set
    predictions, labels, probs, images, indices, filenames, classes = evaluate_model(model, test_loader, dataset, device)
    
    # Get TP and TN samples
    correct_samples = get_tp_tn_samples(predictions, labels, images, indices, filenames, classes, num_samples)
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Process TP samples
    tp_orig_images = []
    tp_saliency_maps = []
    tp_gradcam_heatmaps = []
    tp_image_labels = []
    tp_image_filenames = []
    tp_image_classes = []
    
    for img, label, idx, filename, class_folder in correct_samples['tp']:
        input_img = img.unsqueeze(0).to(device)
        
        # Saliency Map
        input_sal = input_img.clone().detach().requires_grad_(True)
        output = model(input_sal)
        output_scalar = output[0]
        model.zero_grad()
        output_scalar.backward()
        saliency = input_sal.grad.data.abs()
        saliency, _ = torch.max(saliency, dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        
        # Grad-CAM
        gradcam_raw = grad_cam.generate_cam(input_img)
        
        # Denormalize original image
        orig_img = input_img.squeeze().detach().cpu().numpy()
        orig_img = np.transpose(orig_img, (1, 2, 0))
        orig_img = (orig_img * 0.5) + 0.5
        orig_img = np.clip(orig_img, 0, 1)
        
        tp_orig_images.append(orig_img)
        tp_saliency_maps.append(saliency)
        tp_gradcam_heatmaps.append(gradcam_raw)
        tp_image_labels.append(label.item())
        tp_image_filenames.append(extract_seq_number(filename))
        tp_image_classes.append(class_folder)
    
    # Process TN samples
    tn_orig_images = []
    tn_saliency_maps = []
    tn_gradcam_heatmaps = []
    tn_image_labels = []
    tn_image_filenames = []
    tn_image_classes = []
    
    for img, label, idx, filename, class_folder in correct_samples['tn']:
        input_img = img.unsqueeze(0).to(device)
        
        # Saliency Map
        input_sal = input_img.clone().detach().requires_grad_(True)
        output = model(input_sal)
        output_scalar = output[0]
        model.zero_grad()
        output_scalar.backward()
        saliency = input_sal.grad.data.abs()
        saliency, _ = torch.max(saliency, dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        
        # Grad-CAM
        gradcam_raw = grad_cam.generate_cam(input_img)
        
        # Denormalize original image
        orig_img = input_img.squeeze().detach().cpu().numpy()
        orig_img = np.transpose(orig_img, (1, 2, 0))
        orig_img = (orig_img * 0.5) + 0.5
        orig_img = np.clip(orig_img, 0, 1)
        
        tn_orig_images.append(orig_img)
        tn_saliency_maps.append(saliency)
        tn_gradcam_heatmaps.append(gradcam_raw)
        tn_image_labels.append(label.item())
        tn_image_filenames.append(extract_seq_number(filename))
        tn_image_classes.append(class_folder)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
    
    # Create separate figures for TP and TN
    tp_output_path = output_path_base.replace('.png', '_TP.png')
    tn_output_path = output_path_base.replace('.png', '_TN.png')
    
    # Create TP figure
    tp_data = (tp_orig_images, tp_saliency_maps, tp_gradcam_heatmaps, 
               tp_image_filenames, tp_image_classes, ['TP'] * len(tp_orig_images))
    create_visualization_figure(tp_data, "TP", tp_output_path)
    
    # Create TN figure
    tn_data = (tn_orig_images, tn_saliency_maps, tn_gradcam_heatmaps,
               tn_image_filenames, tn_image_classes, ['TN'] * len(tn_orig_images))
    create_visualization_figure(tn_data, "TN", tn_output_path)
    
    print(f"Created separate TP ({len(tp_orig_images)}/{correct_samples['tp_count']} samples) and TN ({len(tn_orig_images)}/{correct_samples['tn_count']} samples) figures.")

def main():
    set_seed(42)
    
    # Model configurations with correct model paths from dna_image_saved_models
    models_config = {
        'fcgr_arab_acc': 'dna_image_saved_models/resnet50_fcgr_arab_acc_20250415_185111.pth',
        'fcgr_arab_don': 'dna_image_saved_models/resnet50_fcgr_arab_don_20250415_214118.pth',
        'fcgr_homo_acc': 'dna_image_saved_models/resnet50_fcgr_homo_acc_20250416_003324.pth',
        'fcgr_homo_don': 'dna_image_saved_models/resnet50_fcgr_homo_don_20250416_032643.pth',
        'fixed_arab_acc': 'dna_image_saved_models/resnet50_fixed_arab_acc_20250415_184840.pth',
        'fixed_arab_don': 'dna_image_saved_models/resnet50_fixed_arab_don_20250415_234130.pth',
        'fixed_homo_acc': 'dna_image_saved_models/resnet50_fixed_homo_acc_20250416_043420.pth',
        'fixed_homo_don': 'dna_image_saved_models/resnet50_fixed_homo_don_20250416_151039.pth'
    }
    
    # Dataset paths with correct test image directories
    dataset_paths = {
        'fcgr_arab_acc': 'Test_Image_fcgr/arab_acc',
        'fcgr_arab_don': 'Test_Image_fcgr/arab_don',
        'fcgr_homo_acc': 'Test_Image_fcgr/homo_acc',
        'fcgr_homo_don': 'Test_Image_fcgr/homo_don',
        'fixed_arab_acc': 'Test_Image_fixed/arab_acc',
        'fixed_arab_don': 'Test_Image_fixed/arab_don',
        'fixed_homo_acc': 'Test_Image_fixed/homo_acc',
        'fixed_homo_don': 'Test_Image_fixed/homo_don'
    }
    
    # Data transforms - exactly matching training script
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create final_figures directory
    os.makedirs("final_figures", exist_ok=True)
    
    # Generate visualizations for each dataset
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"\nProcessing ResNet50 on {dataset_name} dataset...")
        
        try:
            # Load dataset
            dataset = datasets.ImageFolder(dataset_path, transform=transform)
            test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
            
            # Load model
            model_path = models_config[dataset_name]
            model, target_layer = load_model(model_path)
            
            # Generate visualizations
            output_path = f"final_figures/resnet50_{dataset_name}_visualizations.png"
            generate_visualizations(
                model,
                target_layer,
                test_loader,
                dataset,
                output_path,
                dataset_name,
                num_samples=3  # 3 samples per species as requested
            )
            
            # Clear memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing ResNet50 on {dataset_name} dataset: {str(e)}")

if __name__ == "__main__":
    main()
