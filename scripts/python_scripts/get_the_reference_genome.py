import argparse
import os
from Bio import Entrez

def get_the_reference_genome(assembly_id, email, output_folder_id):

    output_path = os.path.join("data/samples/", f"sample_{output_folder_id}")
    os.makedirs(output_path, exist_ok = True)

    genome_handle = Entrez.efetch(db = "nucleotide", id = assembly_id, rettype = "fasta", retmode = "text")
    genome_data = genome_handle.read()
    genome_handle.close()

    annotation_handle = Entrez.efetch(db = "nucleotide", id = assembly_id, rettype = "gtf", retmode = "text")
    annotation_data = annotation_handle.read()
    annotation_handle.close()

    with open(os.path.join(output_path, f"reference_genome_{assembly_id}.fasta"), "w") as genome_file:
        genome_file.write(genome_data)

    with open(os.path.join(output_path, f"reference_genome_{assembly_id}.gtf"), "w") as annotation_file:
        annotation_file.write(annotation_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Download reference genome and annotation")
    parser.add_argument("assembly_id", type = str, help = "NCBI GenBank ID")
    parser.add_argument("email", type = str, help = "Your email address")
    parser.add_argument("output_folder_id", type = int, help = "Output folder id to save the data")
    args = parser.parse_args()

    Entrez.email = args.email
    
    get_the_reference_genome(args.assembly_id, args.email, args.output_folder_id)

