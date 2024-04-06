import argparse
import numpy as np

from keras.models import load_model
from keras.utils import to_categorical
from Models.PGAMv2.get_training_data import get_training_data
from Models.find_files import find_files

nucleotide_codes = {
    'A': 1, 'C': 2, 'G': 3, 'T': 4,
    # 'R': 5, 'Y': 6, 'S': 7, 'W': 8,
    # 'K': 9, 'M': 10, 'B': 11, 'D': 12,
    # 'H': 13, 'V': 14,
    'N': 5
}

def detect(seq):
    with open('output.gtf', 'w') as file:
        file.write("# PGAM")
        file.write("\n# PGAMv2")
        
        exon_start = None
        exon_end = None
        exon_num = 1
        gap_length = 0
        
        for i, value in enumerate(seq):
            if value == 1:
                if exon_start is None:
                    exon_start = i
                exon_end = i + 1
                gap_length = 0
            else:
                gap_length += 1
                if exon_start is not None and (gap_length >= 10 or i == len(seq) - 1):
                    if exon_end - exon_start >= 10:
                        file.write(f"\nBA000007.3\tPGAM\ttranscript\t{exon_start}\t{exon_end}\t1000\t.\t.\tgene_id \"PGAM.{exon_num}\"; transcript_id \"PGAM.{exon_num}.1\";")
                        file.write(f"\nBA000007.3\tPGAM\texon\t{exon_start}\t{exon_end}\t1000\t.\t.\tgene_id \"PGAM.{exon_num}\"; transcript_id \"PGAM.{exon_num}.1\"; exon_number \"1\";")
                        exon_num += 1
                    else:
                        file.write(f" {exon_end}\t.\t.\tgene_id \"PGAM.{exon_num - 1}\"; transcript_id \"PGAM.{exon_num - 1}.1\";")
                    exon_start = None
                    exon_end = None
                    gap_length = 0 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Detect PGAMv2.")
    parser.add_argument("n_window", type = int ,help = "The number of nucleotides (window size) up to the current nucleotide in the prediction")
    parser.add_argument("n_samples_per_epoch", type = int, help = "The number of samples that will be trained pack by pack")

    args = parser.parse_args()

    fasta_test, gtf_test = find_files('./', 'test_sample', 'test_sample')

    predictions = []
    # y_true = []

    model = load_model('./Models/PGAMv2/reports/PGAMv2.h5')
    model.summary()

    for k, (X_feature, y_target) in enumerate(get_training_data(fasta_test, gtf_test, args.n_window, args.n_samples_per_epoch, nucleotide_codes)):
        predicted = model.predict(X_feature)

        predictions.append(predicted)
        # y_true.append(to_categorical(y_target, num_classes = 2))


    predictions = np.concatenate(predictions)
    # y_true = np.concatenate(y_true)

    detect(predictions)

