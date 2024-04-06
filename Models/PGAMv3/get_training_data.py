import numpy as np
from Models.read_fasta_file import read_fasta_file
from Models.parse_gtf_file import parse_gtf_file
from Models.encode_sequence import encode_sequence

def get_training_data(fasta_files, gtf_files, n_window, max_len, nucleotide_codes):
    X_pre = []
    X_next = []
    y = []
    for sequence, exon in zip(fasta_files, gtf_files):
        sequence = str(read_fasta_file(sequence))
        exon = parse_gtf_file(exon)
            
        for j in range(0, len(sequence)):
            start = max(0, j - n_window + 1)
            end = min(len(sequence), j + 1)

            window = encode_sequence(sequence[start : end], nucleotide_codes)

            pad_start = max(0, n_window - len(window))

            if pad_start > 0:
                padding = [nucleotide_codes['N']] * pad_start
                window = padding + list(window)

            X_pre.append(np.array(window))

            start = j + 1
            end = min(j + n_window + 1, len(sequence))

            window = sequence[start : end]

            pad_start = max(0, n_window - len(window))

            if pad_start > 0:
                window = np.concatenate(window, (np.array([nucleotide_codes['N']] * pad_start)))

            exon_val = 1 if any(start <= j < end for transcript_exons in exon.values() for start, end in transcript_exons) else 0

            X_next.append(np.array(window))
            y.append(exon_val)

            if len(X_pre) >= max_len or j == len(exon) - 1:
                yield np.array(X_pre), np.array(X_next), np.array(y)
                X_pre.clear()
                X_next.clear()
                y.clear()

