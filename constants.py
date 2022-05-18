
# DATA
dilutions = ['0001', '0002', '0004', '0008', '0016', '0032', '0064']
aas = ['Proline', 'Asparagine', 'Lysine', 'Phenylalanine', 'Tyrosine']
pps = ['Cytosine', 'Uracil', 'Thymine', 'Adenine']

import pandas
data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/filtered_data_with_mz.csv', index_col=0).iloc[:, :2]
mz_dict = {k: v for k, v in zip(data['name'], data['mz'].astype('float'))}

# ML
seed = 2022
split_ratio = 0.8

n_jobs = 8
save_to = '/Users/andreidm/ETH/projects/calibration/res/'