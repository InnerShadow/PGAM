import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def parse_cigar_file(cigar_file):
    {'M': [], 'I': [], 'D': [], 'N': [], 'S': [], 'H': [], 'P': [], '=': [], 'X': []}
    cigar_M = []
    cigar_I = []
    cigar_D = []
    cigar_N = []
    cigar_S = []
    cigar_H = []
    cigar_P = [] 
    cigar_X = []

    with open(cigar_file, "r") as file:
        for line in file:
            cigar = line.strip()
            if cigar:
                cigar_operations = [(int(''.join(filter(str.isdigit, op))), op) for op in cigar.split("M") if op]
                for op_len, op_type in cigar_operations:
                    cigar_counts[op_type].append(op_len)

    return cigar_counts

def plot_boxplot(cigar_counts, output_png):
    df = pd.DataFrame(cigar_counts)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.xlabel('CIGAR Operation')
    plt.ylabel('Count')
    plt.title('Distribution of CIGAR Operations')
    plt.savefig(output_png)  # Save box plot as PNG
    plt.show()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description = "Generate box plot for CIGAR operations.")
    # parser.add_argument("cigar_file", help = "Path to the file containing CIGAR strings")
    # parser.add_argument("output_png", help = "Path to save the box plot PNG file")

    # args = parser.parse_args()

    # cigar_file = args.cigar_file
    # output_png = args.output_png

    cigar_file = "ERR12100549_CIGAR_info.txt"
    output_png = "tt.png"

    cigar_counts = parse_cigar_file(cigar_file)
    plot_boxplot(cigar_counts, output_png)
