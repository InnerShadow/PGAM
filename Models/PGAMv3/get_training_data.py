import numpy as np

def get_training_data(encoded_sequences_array, exon_array, n_window, max_len, nucleotide_codes):
    X_pre = []
    X_next = []
    y = []
    for sequence, exon in zip(encoded_sequences_array, exon_array):
        for j in range(0, len(exon)):
            start = max(0, j - n_window)
            end = j

            window = sequence[start:end]

            pad_start = max(0, n_window - len(window))

            if pad_start > 0:
                window = np.concatenate((np.array([nucleotide_codes['N']] * pad_start), window))

            X_pre.append(window)

            start = j
            end = min(j + n_window, len(exon_array))

            window = sequence[start:end]

            pad_start = max(0, n_window - len(window))

            if pad_start > 0:
                window = np.concatenate(window, (np.array([nucleotide_codes['N']] * pad_start)))

            X_next.append(window)
            y.append(exon[j])

            if len(X_pre) >= max_len or j == len(exon) - 1:
                yield np.array(X_pre), np.array(X_next), np.array(y)
                X_pre.clear()
                X_next.clear()
                y.clear()

