# SpliceImage

A DNA Sequence Imaging Approach to Predict Splice Sites Using Deep Learning.

## Project Overview

SpliceImage uses image-based representations of DNA sequences (Frequency Chaos Game Representation - FCGR and Dinucleotide Fixed Color Pattern - DFCP) to convert DNA sequences into images, which are then analyzed using deep learning models (ResNet50) to predict splice sites. The project includes tools for training models, visualizing DNA sequences, and interpreting model predictions through saliency maps and Grad-CAM visualizations.

## Key Features

- Conversion of DNA sequences to FCGR and DFCP images for visual representation
- Training and evaluation of ResNet50 models for splice site prediction
- Analysis of both acceptor and donor splice sites
- Support for multiple species (Arabidopsis thaliana and Homo sapiens)
- Visualization of model attention using saliency maps and Grad-CAM
- Identification and analysis of important 6-mers (hexamers) in splice site recognition
- Comprehensive evaluation metrics and result storage

## Main Components

### Data Conversion Scripts
- `dna_image_convert_FCGR.py`: Converts DNA sequences to Frequency Chaos Game Representation images
- `dna_image_convert_DFCP.py`: Converts DNA sequences to Dinucleotide Fixed Color Pattern images

### Training and Evaluation
- `train.py`: Trains ResNet50 models on DNA sequence image datasets
- `evaluate_models_test_dirs_homo.py`: Evaluates trained models on Homo sapiens test data

### Interpretability and Visualization
- `interpretability_FCGR.py`: Analysis and visualization of model predictions for FCGR images
- `interpretability_DCFP.py`: Analysis and visualization of model predictions for DFCP images
- `color_label.py`: Generates visualization of the DNA dinucleotide color mapping used in representations

## Data Requirements

The code expects data organized in the following structure:
- Training and testing datasets for each species
- Separate datasets for acceptor and donor sites
- Each dataset containing positive and negative examples of splice sites

## Dependencies

- Python 3.6+
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL (Pillow)
- tqdm
- seaborn

## Usage

### Converting DNA Sequences to Images

```bash
python dna_image_convert_FCGR.py  # For FCGR representation
python dna_image_convert_DFCP.py  # For DFCP representation
```

### Training Models

```bash
python train.py
```

### Evaluating Models

```bash
python evaluate_models_test_dirs_homo.py
```

### Analyzing and Visualizing Results

```bash
python interpretability_FCGR.py  # For FCGR-based analysis
python interpretability_DCFP.py  # For DFCP-based analysis
```

### Generating Color Label Reference

```bash
python color_label.py
```

## Results

The analysis produces:
- Model evaluation metrics (accuracy, precision, recall, F1 score)
- Visualizations of DNA sequences using FCGR and DFCP methods
- Saliency maps and Grad-CAM visualizations highlighting important regions
- CSV files with the top 10 important 6-mers for each sequence
- Combined result files for comprehensive analysis

## Citation

To be added later.

## License

MIT