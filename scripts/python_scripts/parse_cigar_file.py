import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def parse_cigar_file(cigar_file):
    cigar_counts = {'M': 0, 'I': 0, 'D': 0, 'N': 0, 'S': 0}

    with open(cigar_file, "r") as file:
        for line in file:
            cigar = line.strip()
            if cigar:
                string = ""
                for c in cigar:
                    if c.isdigit():
                        string += c
                    else:
                        cigar_counts[c] += int(string)
                        string = ""

    return cigar_counts


def plot_pie_chart(cigar_file, cigar_counts, output_png):
    labels = list(cigar_counts.keys())
    values = list(cigar_counts.values())

    colors = sns.color_palette('pastel')[0:len(labels)]

    plt.figure(figsize = (10, 6))
    plt.pie(values, labels = labels, autopct = '%1.1f%%', startangle = 140, colors = colors, wedgeprops = {'edgecolor': 'black'})
    plt.title(f"CIGAR Operation Distribution for {cigar_file.split('_')[0]}", fontsize = 16)
    plt.axis('equal')
    plt.legend(loc = "best", fontsize=12)
    plt.savefig(output_png)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate pie chart for CIGAR operations.")
    parser.add_argument("cigar_file", help = "Path to the file containing CIGAR strings")
    parser.add_argument("output_png", help = "Path to save the pie chart PNG file")

    args = parser.parse_args()

    cigar_file = args.cigar_file
    output_png = args.output_png

    cigar_counts = parse_cigar_file(cigar_file)
    plot_pie_chart(cigar_file, cigar_counts, output_png)

