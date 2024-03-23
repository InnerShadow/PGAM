import os
from glob import glob

def find_files(directory, dirr):
    fasta_files = []
    gtf_files = []

    for subdir in glob(os.path.join(directory, 'data', dirr, 'sample_*')):
        fasta_file = glob(os.path.join(subdir, '*.fasta'))
        gtf_file = glob(os.path.join(subdir, '*_gtf_merged.gtf'))

        if fasta_file:
            fasta_files.extend(fasta_file)
        if gtf_file:
            gtf_files.extend(gtf_file)

    return fasta_files, gtf_files

