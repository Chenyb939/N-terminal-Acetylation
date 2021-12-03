import os
import re
import pandas as pd
from Bio import SwissProt


def get_data(filename, outdir, sign):
    list = ["gname", "gsets", "gsete", "gann", "gseq"]
    df = pd.DataFrame(columns=list)
    non_list = ['transferase', 'amidase', 'O-', 'non-acetylated', 'glucosamine', 'binding', 'pyrophosphorylase',
                'ligase', 'etherase', 'deacetylase', 'transporter', 'Aspirin', 'minidase', 'Probable', 'epimerase',
                'Putative', 'reductase', 'Di-N-acetylchitobiase', 'monooxygenase', 'Enzyme', 'Key residue',
                'dehydrogenase', 'kinase', 'lectin', 'synthase', 'ARLIAK', 'transcarbamylase', 'proteasomes',
                'hydroxylase', 'May prevent', 'Alters', 'efficiency', 'Increases', 'decrease', 'GALAC3', 'pv18',
                'amide nitrogen', 'Microbial infection', 'ATG7', 'sulfatase', 'oxidase', 'hydrolase', 'lyase',
                'dipeptidase', 'transaminase', 'de-N-acetylase', 'synthetase', 'phosphorylase', 'phosphatase',
                'Inactive', 'phosphatidylinositol', 'EIIBC', 'transacetylase', 'IIC', 'IID', 'S-', 'Not',
                'Prevents', 'Decreased', 'Reduces', 'effect', 'acetylneuraminate', 'acetylglutamate', 'glutamate', 'galactosamine']
    with open(filename) as handle:
        records = SwissProt.parse(handle)
        for record in records:
            gann = record.features
            for i in range(len(gann)):

                ann = gann[i].qualifiers['description']
                pattern = re.compile(sign)
                if pattern.search(ann) == None:
                    continue
                else:
                    if any(str in ann for str in non_list):
                        pass
                    else:
                        gname = ''
                        for name in record.accessions:
                            gname += str(name) + ','
                        a = record.features[i].location.nofuzzy_start
                        gsets = record.features[i].location.nofuzzy_start + 1
                        gsete = record.features[i].location.nofuzzy_end
                        gseq = record.sequence
                        d_dict = {"gname": gname, "gsets": gsets, "gsete": gsete, "gann": ann, "gseq": gseq}
                        df = df.append(d_dict, ignore_index=True)
        df.to_csv(os.path.join(outdir, 'data.csv'), index=False)


if __name__ == '__main__':
    # get_data("/home/chenyb/PTM/data/uniport/2015/data.dat", '/home/chenyb/PTM/data/uniport/2015/', 'N?-acetyl')
    get_data("E:/hefei/PTM/data/2015/uniprot_sprot-only2015_04/uniprot_sprot_dat/uniprot_sprot.dat", 'E:/hefei/PTM/data/2015/', 'N?-acetyl')
