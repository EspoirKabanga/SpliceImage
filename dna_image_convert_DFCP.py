import numpy as np
from PIL import Image
import os


def generate_fixed_dinucleotide_colors():
    return {
        'AA': [255, 0, 0], 'AC': [255, 128, 0], 'AG': [255, 255, 0], 'AT': [128, 255, 0],
        'CA': [0, 255, 0], 'CC': [0, 255, 128], 'CG': [0, 255, 255], 'CT': [0, 128, 255],
        'GA': [0, 0, 255], 'GC': [128, 0, 255], 'GG': [255, 0, 255], 'GT': [255, 0, 128],
        'TA': [128, 128, 128], 'TC': [128, 64, 0], 'TG': [64, 128, 128], 'TT': [192, 192, 192]
    }


dinucleotide_colors = generate_fixed_dinucleotide_colors()
default_color = [255, 255, 255]


def dna_to_dinucleotide_image(dna_sequence, name, output_path):
    sequence_length = len(dna_sequence)

    dna_array = np.array(list(dna_sequence))

    seq_grid_x, seq_grid_y = np.meshgrid(dna_array, dna_array, indexing='ij')
    dinucleotide_grid = np.char.add(seq_grid_x, seq_grid_y)

    image_grid = np.full((sequence_length, sequence_length, 3), default_color, dtype=np.uint8)

    for dinuc, color in dinucleotide_colors.items():
        matches = dinucleotide_grid == dinuc
        image_grid[matches] = color

    img = Image.fromarray(image_grid, 'RGB')
    img.save(f'{output_path}{name}.png')


if __name__ == "__main__":
    with open('DRANet/arabidopsis_acceptor_negative.txt') as file:
        data = file.readlines()

    output_path = "Test_Image_fixed/arab_acc/neg/"
    os.makedirs(output_path, exist_ok=True)

    print("Total sequences:", len(data))

    for count, line in enumerate(data[30001:35001], start=1): # 30000
        dna_seq = line.strip()
        dna_to_dinucleotide_image(dna_seq, f'seq_{count+30000}', output_path)  # delete 30000

        if count % 100 == 0:
            print(f"Processed {count} sequences")

    print("---CONVERSION COMPLETED SUCCESSFULLY---")