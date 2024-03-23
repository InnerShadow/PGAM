def parse_gtf_file(gtf_file):
    exons = {}

    with open(gtf_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                cols = line.strip().split('\t')
                feature_type = cols[2]
                if feature_type == 'exon':
                    chromosome = cols[0]
                    start = int(cols[3])
                    end = int(cols[4])
                    gene_id = cols[8].split(';')[0].split('"')[1]
                    if gene_id not in exons:
                        exons[gene_id] = []
                    exons[gene_id].append((start, end))

    return exons

