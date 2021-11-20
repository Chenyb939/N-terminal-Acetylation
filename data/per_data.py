import os
import re
import pandas as pd


def load_data(filename, splitsign='\t'):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                pass
            else:
                data.append(line.split(splitsign))
    del lines, line
    return data


def get_data(data, outdir, sign):
    non_list = ['transferase', 'amidase', 'O-', 'non-acetylated', 'glucosamine', 'binding', 'pyrophosphorylase',
                'ligase', 'etherase', 'deacetylase', 'transporter', 'Aspirin', 'minidase', 'Probable', 'epimerase',
                'Putative', 'reductase', 'Di-N-acetylchitobiase', 'monooxygenase', 'Enzyme', 'Key residue',
                'dehydrogenase', 'kinase', 'lectin', 'synthase', 'ARLIAK', 'transcarbamylase', 'proteasomes',
                'hydroxylase', 'May prevent', 'Alters', 'efficiency', 'Increases', 'decrease', 'GALAC3', 'pv18',
                'amide nitrogen', 'Microbial infection', 'ATG7', 'sulfatase', 'oxidase', 'hydrolase', 'lyase',
                'dipeptidase', 'transaminase', 'de-N-acetylase', 'synthetase', 'phosphorylase', 'phosphatase',
                'Inactive', 'phosphatidylinositol', 'EIIBC', 'transacetylase', 'IIC', 'IID']
    list = ["gname", "gsets", "gsete", "gann"]
    df = pd.DataFrame(columns=list)
    for i in range(len(data)):
        if i % 100000 == 0:
            print(str(i) + ' items has been proceed.')
        gann = data[i][8]
        pattern = re.compile(sign)
        if pattern.search(gann) == None:
            continue
        else:
            if any(str in gann for str in non_list):
                pass
            else:
                gname = data[i][0]
                gsets = data[i][3]
                gsete = data[i][4]
                d_dict = {"gname": gname, "gsets": gsets, "gsete": gsete, "gann": gann}
                df = df.append(d_dict, ignore_index=True)
    df.to_csv(os.path.join(outdir, 'data.csv'), index=False)
    return df


if __name__ == '__main__':
    data = load_data('/home/chenyb/PTM/data/uniport/2015/data', '\t')
    data_df = get_data(data, '/home/chenyb/PTM/data/uniport/2015/', 'N?-acetyl')

# 2015
# https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2015_04/knowledgebase/knowledgebase2015_04.tar.gz
