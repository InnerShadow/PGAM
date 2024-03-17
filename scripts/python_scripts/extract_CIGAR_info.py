import argparse
import pysam

def extract_CIGAR_info(input_bam, output_txt):
    with pysam.AlignmentFile(input_bam, "rb") as bam:
        with open(output_txt, "w") as output:
            for read in bam:
                sigar = read.cigarstring
                output.write(sigar + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Extract CIGAR information from a BAM file.")
    parser.add_argument("input_bam", help = "Input BAM file")
    parser.add_argument("output_txt", help = "Output text file")

    args = parser.parse_args()

    input_bam = args.input_bam
    output_txt = args.output_txt

    extract_CIGAR_info(input_bam, output_txt)

    print("CIGAR information has been successfully extracted and written to", output_txt)

