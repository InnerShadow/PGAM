import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_info(file_path):
    alignment_data = {'aligned 0 times': 0, 
                      'exactly 1 time' : 0, 
                      'aligned >1 times' : 0, 
                      'overall alignment rate' : 0}
    
    with open(file_path, 'r') as file:
        data = file.read()

        matches = re.findall(r'(\d+\.\d+)%.*aligned 0 times', data)
        alignment_data['aligned 0 times'] = float(matches[0])

        matches = re.findall(r'(\d+\.\d+)%.*aligned exactly 1 time', data)
        alignment_data['exactly 1 time'] = float(matches[0])

        matches = re.findall(r'(\d+\.\d+)%.*aligned >1 times', data)
        alignment_data['aligned >1 times'] = float(matches[0])

        matches = re.findall(r'(\d+\.\d+)% overall alignment rate', data)
        alignment_data['overall alignment rate'] = float(matches[0])

        return pd.DataFrame(pd.Series(alignment_data)).T


def find_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_alignments_report.txt'):
                yield os.path.join(root, file)


def find_full_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_alignments_report.txt'):
                yield root + file


if __name__ == '__main__':
    all_alignment_data = pd.DataFrame()
    outliers = []

    for file_path in find_files("data/samples/"):
        alignment_data = extract_info(file_path)
        all_alignment_data = pd.concat([all_alignment_data, alignment_data])


    for i, file_path in enumerate(find_full_files("data/samples/")):
        for j in ['aligned >1 times', 'overall alignment rate']:
            q1 = all_alignment_data[j].quantile(0.25)
            q3 = all_alignment_data[j].quantile(0.75)
            if float(all_alignment_data[j].iloc[i]) < q1 - 1.5 * (q3 - q1):
                outliers.append(f'{file_path} ---------- { j }')


    outliers.append("")

    for i, file_path in enumerate(find_full_files("data/samples/")):
        for j in ['aligned 0 times', 'exactly 1 time']:
            q1 = all_alignment_data[j].quantile(0.25)
            q3 = all_alignment_data[j].quantile(0.75)
            if float(all_alignment_data[j].iloc[i]) > q3 + 1.5 * (q3 - q1):
                outliers.append(f'{file_path} ---------- { j }')


    with open("outliers.txt", "w") as f:
        for outlier in outliers:
            f.write(outlier + "\n")


    for i in all_alignment_data.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=all_alignment_data[i])
        plt.title('Alignment Percentages')
        plt.xlabel(f'Number of samples that {i}')
        plt.ylabel('Percentage')
        plt.savefig(f"{i}_all_samples_boxplot.png")
        plt.close()

