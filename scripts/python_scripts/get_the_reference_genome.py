import argparse
import os
from Bio import Entrez

def get_the_reference_genome(assembly_id, email):
    Entrez.email = email

    output_path = os.path.join(".", "")
    os.makedirs(output_path, exist_ok = True)

    genome_handle = Entrez.efetch(db = "nucleotide", id = assembly_id, rettype = "fasta", retmode = "text")
    genome_data = genome_handle.read()
    genome_handle.close()

    annotation_handle = Entrez.efetch(db = "nucleotide", id = assembly_id, rettype = "gff3", retmode = "text")
    annotation_data = annotation_handle.read()
    annotation_handle.close()

    with open(os.path.join(output_path, f"reference_genome_{assembly_id}.fasta"), "w") as genome_file:
        genome_file.write(genome_data)

    with open(os.path.join(output_path, f"reference_genome_{assembly_id}.gff3"), "w") as annotation_file:
        annotation_file.write(annotation_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Download reference genome and annotation")
    parser.add_argument("assembly_id", type = str, help = "NCBI GenBank ID")
    parser.add_argument("email", type = str, help = "Your email address")
    args = parser.parse_args()
    
    get_the_reference_genome(args.assembly_id, args.email)

    print(f"completing the download of {args.assembly_id} files!")

