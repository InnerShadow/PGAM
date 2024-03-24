import numpy as np

def get_training_data(encoded_sequences_array, exon_array, n_window, max_len, nucleotide_codes):
    X = []
    y = []
    for sequence, exon in zip(encoded_sequences_array, exon_array):
        for j in range(0, len(exon)):
            start = max(0, j - n_window)
            end = j

            window = sequence[start:end]

            pad_start = max(0, n_window - len(window))

            if pad_start > 0:
                window = np.concatenate((np.array([nucleotide_codes['N']] * pad_start), window))

            X.append(window)
            y.append(exon[j])

            if len(X) >= max_len or j == len(exon) - 1:
                yield np.array(X), np.array(y)
                X.clear()
                y.clear()

