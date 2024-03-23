
def encode_sequence(sequence, nucleotide_codes):
    encoded_sequence = []
    for base in sequence:
        encoded_sequence.append(nucleotide_codes.get(base.upper(), 0))
    return encoded_sequence

