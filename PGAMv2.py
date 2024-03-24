from Models.encode_sequence import encode_sequence
from Models.find_files import find_files
from Models.parse_gtf_file import parse_gtf_file
from Models.read_fasta_file import read_fasta_file

from Models.PGAMv2.train_model import train_model
from Models.PGAMv2.get_model import get_model

import argparse
import numpy as np

nucleotide_codes = {
    'A': 1, 'C': 2, 'G': 3, 'T': 4,
    # 'R': 5, 'Y': 6, 'S': 7, 'W': 8,
    # 'K': 9, 'M': 10, 'B': 11, 'D': 12,
    # 'H': 13, 'V': 14,
    'N': 5
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Train PGAMv1.")
    parser.add_argument("n_window", type = int ,help = "The number of nucleotides (window size) up to the current nucleotide in the prediction")
    parser.add_argument("n_samples_per_epoch", type = int, help = "The number of samples that will be trained pack by pack")
    parser.add_argument("batch_size", type = int, help = "The size of the mini-butch")
    parser.add_argument("epochs", type = int, help = "The number of epochs")
    parser.add_argument("embedding_size", type = int, help = "The number of outputs neurons on the embending layer.")
    parser.add_argument("n_times", type = int, help = "The number of times that n_samples_per_epoch will fit on model")

    args = parser.parse_args()

    fasta_files, gtf_files = find_files('./', 'samples', 'sample_*')

    sequences = []
    for fasta_file in fasta_files:
        sequences.append(read_fasta_file(fasta_file))


    sequences_array = np.array([list(seq) for seq in sequences], dtype = 'S1')
    
    exons_info = {}
    exon_array = np.zeros_like(sequences_array, dtype = int)

    for i, gtf_file in enumerate(gtf_files):
        exons_info.update(parse_gtf_file(gtf_file))

        for gene_id, exon_positions in exons_info.items():
            for start, end in exon_positions:
                exon_array[i, start - 1 : end] = 1
        # print(exons_info)

        exons_info.clear()


    encoded_sequences = []
    for sequence in sequences:
        encoded_sequences.append(encode_sequence(sequence, nucleotide_codes))

    sequences.clear()
    encoded_sequences_array = np.array(encoded_sequences)

    model = get_model(args.n_window, len(nucleotide_codes) + 1, args.embedding_size)
    model = train_model(model, args.epochs, encoded_sequences_array, exon_array, args.n_window, args.n_samples_per_epoch, args.n_times, args.batch_size, nucleotide_codes)

