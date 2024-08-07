#!/usr/bin/env bash

# Check if correct number of arguments provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <email> <file_path> <numer of thread fro samtools>"
    echo "Set num of thread to 1"
    exit 1
fi

email=$1
file_path=$2

if [$# -eq 3 ]; then
    num_theds=$3
else 
    num_theds=1
fi


# Check if file exists
if [ ! -f "$file_path" ]; then
    echo "File not found: $file_path"
    exit 1
fi

index=0

# Read the file line by line
while IFS= read -r line || [ -n "$line" ]; do
    # If the line is not empty, it's either a reference genome identifier or a raw reads identifier
    if [ -n "$line" ]; then
        if [ -z "$reference_genome" ]; then
            # If reference_genome is empty, this is a reference genome identifier
            reference_genome=$line
        else
            # If reference_genome is not empty, this is a raw reads identifier
            raw_reads+=("$line")
        fi
    else
        # If the line is empty, it's the end of a record
        if [ -n "$reference_genome" ] && [ ${#raw_reads[@]} -gt 0 ]; then
            # Preprocess reference genome
            echo "Reference Genome: $reference_genome"

            # Create directory for this sample
            mkdir -p "data/samples/sample_${index}"
            cd "data/samples/sample_${index}"

            # Grab .fasta & .gff3 file using get_the_reference_genome.py python script
            python3 ../../../scripts/python_scripts/get_the_reference_genome.py "${reference_genome}" "${email}"

            # Make directory gor index files
            mkdir index

            # Index reference genome
            bowtie2-build "reference_genome_${reference_genome}.fasta" "index/${reference_genome}_index"

            echo "Raw Reads:"
            for read_id in "${raw_reads[@]}"; do
                echo "${read_id}"

                # Create directory for this reads 
                mkdir "Read_${read_id}"
                cd "Read_${read_id}"

                # Grab reads 
                prefetch "${read_id}" -o "${read_id}.sra"

                # Get reads quality
                fastq-dump "${read_id}.sra"

                # Make FastQC analysis, save html doc for report
                fastqc "${read_id}.fastq"

                # Remove reads with low quality &
                # discard remaining adapters &
                # generate fastp report in html format 
                fastp -i "${read_id}.fastq" -o "${read_id}_fastp.fastq" -Q -h "${read_id}_fastp_report.html"

                # Align reads to the reference genome & save align results to .txt file
                (bowtie2 --local -p "${num_theds}" -x "../index/${reference_genome}_index" -U "${read_id}_fastp.fastq" \
                    | samtools view -O BAM -b -o "${read_id}_fastp_mapping.bam") 2> "${read_id}_alignments_report.txt"

                # Sort alignments
                samtools sort -@ "${num_theds}" -O BAM -o "${read_id}_fastp_mapping_sorted.bam" "${read_id}_fastp_mapping.bam"

                # Index alignments
                samtools index "${read_id}_fastp_mapping_sorted.bam"

                # Grab CIGAR column from reads & same it
                python3 ../../../../scripts/python_scripts/extract_CIGAR_info.py \
                    "${read_id}_fastp_mapping_sorted.bam" "${read_id}_CIGAR_info.txt"

                # Create pie chart for this CIGAR 
                python3 ../../../../scripts/python_scripts/parse_cigar_file.py \
                    "${read_id}_CIGAR_info.txt" "${read_id}_CIGAR_plot.png"

                # Get annotations based on this reads
                stringtie "${read_id}_fastp_mapping_sorted.bam" -g 0 --conservative -s 5 -o "${read_id}.gtf"

                # Remove a lot of things
                rm "fastp.json"
                rm "${read_id}.sra"
                rm "${read_id}.fastq"
                rm "${read_id}_fastp.fastq"
                rm "${read_id}_CIGAR_info.txt"
                rm "${read_id}_fastp_mapping.bam"
                rm "${read_id}_fastp_mapping_sorted.bam"
                rm "${read_id}_fastp_mapping_sorted.bam.bai"

                cd ..

            done
            echo

            # Merge all reads
            find . -name '*.gtf' > "${reference_genome}_gtf_files.txt"
            stringtie --merge -o "${reference_genome}_gtf_merged.gtf" "${reference_genome}_gtf_files.txt"

            # Remove other tmp files
            rm -R index
            rm "reference_genome_${reference_genome}.gff3"
            rm "reference_genome_${reference_genome}_rRNA_coordinates.bed"

            # Back to source directory
            cd ../../../

            # Increment index
            ((index++))

            # Reset variables for the next record
            unset reference_genome
            unset raw_reads
        fi
    fi
done < "$file_path"

