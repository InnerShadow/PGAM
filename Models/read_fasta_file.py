from Bio import SeqIO

def read_fasta_file(fasta_file):
    sequence = ""

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence += str(record.seq)

    return sequence

