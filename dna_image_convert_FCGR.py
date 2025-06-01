import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def generate_fcgr_colormap():
    """Generate a colormap for FCGR visualization."""
    # Creating a colormap from blue to yellow
    cmap = plt.cm.viridis
    return cmap


def get_nucleotide_index(nucleotide):
    """Map nucleotides to indices for FCGR positioning."""
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


def dna_to_fcgr_image(dna_sequence, name, k=6, output_path="dna_image_fcgr/", normalize=True):
    """
    Transform a DNA sequence into a Frequency Chaos Game Representation (FCGR) image.
    
    Parameters:
    -----------
    dna_sequence : str
        The DNA sequence to transform
    name : str
        The name for the output file
    k : int
        The order of FCGR (k-mer length to consider)
    output_path : str
        Directory to save the output image
    normalize : bool
        Whether to normalize frequency counts
    """
    # Handle ambiguous nucleotides by replacing them
    dna_sequence = ''.join(['N' if n not in 'ACGT' else n for n in dna_sequence])
    
    # Initialize the FCGR matrix (2^k x 2^k)
    matrix_size = 2**k
    fcgr_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    
    # Compute FCGR
    for i in range(len(dna_sequence) - k + 1):
        kmer = dna_sequence[i:i+k]
        
        # Skip k-mers with ambiguous nucleotides
        if 'N' in kmer:
            continue
            
        # Calculate position in FCGR matrix
        x, y = 0, 0
        for j, nuc in enumerate(kmer):
            nuc_idx = get_nucleotide_index(nuc)
            if nuc_idx == -1:  # Skip if ambiguous nucleotide
                break
                
            # Update position based on binary representation
            bit_x = (nuc_idx & 1)  # Least significant bit
            bit_y = ((nuc_idx >> 1) & 1)  # Most significant bit
            
            # Update coordinates (each nucleotide contributes to position)
            x += bit_x * (2**(k-j-1))
            y += bit_y * (2**(k-j-1))
        else:
            # Increment count at calculated position
            fcgr_matrix[y, x] += 1
    
    # Normalize if requested
    if normalize and np.max(fcgr_matrix) > 0:
        fcgr_matrix = fcgr_matrix / np.max(fcgr_matrix)
    
    # Convert to RGB image
    cmap = generate_fcgr_colormap()
    colored_fcgr = cmap(fcgr_matrix)
    
    # Convert to 8-bit RGB
    rgb_image = (colored_fcgr[:, :, :3] * 255).astype(np.uint8)
    
    # Create and save the image
    img = Image.fromarray(rgb_image, 'RGB')
    img.save(f'{output_path}{name}.png')
    
    return fcgr_matrix


# Example usage
if __name__ == "__main__":
    # Example usage like in the original code
    with open('DRANet/arabidopsis_donor_negative.txt') as file:
        data = file.readlines()

    print("Total sequences:", len(data))

    # Process the first 10000 sequences with both methods
    import os
    
    # Create output directories if they don't exist
    os.makedirs("Test_Image_fcgr/arab_don/neg/", exist_ok=True)
    # os.makedirs("dna_image_fcgr_alt/negative/", exist_ok=True)
    
    for count, line in enumerate(data[30001:35001], start=1):
        dna_seq = line.strip()
        # Use the standard implementation
        dna_to_fcgr_image(dna_seq, f'seq_{count+30000}', k=6, output_path="Test_Image_fcgr/arab_don/neg/")
        # Use the alternative implementation
        
        if count % 100 == 0:
            print(f"Processed {count} sequences")