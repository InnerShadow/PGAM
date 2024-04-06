from Models.find_files import find_files

from Models.PGAMv3.train_model import train_model
from Models.PGAMv3.get_model import get_model

import argparse

nucleotide_codes = {
    'A': 1, 'C': 2, 'G': 3, 'T': 4,
    # 'R': 5, 'Y': 6, 'S': 7, 'W': 8,
    # 'K': 9, 'M': 10, 'B': 11, 'D': 12,
    # 'H': 13, 'V': 14,
    'N': 5
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Train PGAMv3.")
    parser.add_argument("n_window", type = int ,help = "The number of nucleotides (window size) up to the current nucleotide in the prediction")
    parser.add_argument("n_samples_per_epoch", type = int, help = "The number of samples that will be trained pack by pack")
    parser.add_argument("batch_size", type = int, help = "The size of the mini-butch")
    parser.add_argument("epochs", type = int, help = "The number of epochs")
    parser.add_argument("embedding_size", type = int, help = "The number of outputs neurons on the embending layer.")
    parser.add_argument("n_times", type = int, help = "The number of times that n_samples_per_epoch will fit on model")

    args = parser.parse_args()

    fasta_files, gtf_files = find_files('./', 'samples', 'sample_*')

    model = get_model(args.n_window, len(nucleotide_codes) + 1, args.embedding_size)
    model = train_model(model, args.epochs, fasta_files, gtf_files, args.n_window, args.n_samples_per_epoch, args.n_times, args.batch_size, nucleotide_codes)

