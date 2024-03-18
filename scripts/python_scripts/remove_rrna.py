# Simpe rypthon script to remove rRNA from aligned .bam file
# Same results (rRNA position in the genome) in .bed file 

import argparse
import pysam

def remove_rrna_reads(bam_file, bed_file, output_bam, offset = 60):
    bam = pysam.AlignmentFile(bam_file, "rb")
    
    rrna_regions = []
    with open(bed_file, 'r') as bed:
        for line in bed:
            fields = line.strip().split('\t')
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            rrna_regions.append((chrom, max(0, start - offset), end + offset))


    out_bam = pysam.AlignmentFile(output_bam, "wb", template=bam)
    
    for read in bam.fetch():
        if not any(start <= read.reference_start < end or start < read.reference_end <= end for chrom, start, end in rrna_regions):
            out_bam.write(read)
    
    bam.close()
    out_bam.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Remove rRNA reads from a BAM file')
    parser.add_argument('bam_file', type = str, help = 'Input BAM file')
    parser.add_argument('bed_file', type = str, help = 'Input BED file with rRNA coordinates')
    parser.add_argument('output_bam', type = str, help = 'Output BAM file')
    parser.add_argument('--offset', type = int, default = 60, help = 'Offset value for rRNA regions (default: 60)')
    args = parser.parse_args()

    remove_rrna_reads(args.bam_file, args.bed_file, args.output_bam, args.offset)

    print("rRNA was removed!")

