
import pandas, seaborn, random
from matplotlib import pyplot
from constants import dilutions, aas, pps, mz_dict


def get_data(path, sample_codes, metabolites):

    data = pandas.read_csv(path, index_col=0).T.sort_index()

    samples = []
    for name in data.index:
        if str(name).split('_')[2] in dilutions and '_'.join(str(name).split('_')[:2]) in sample_codes:
            samples.append(name)

    data = data.loc[samples, metabolites]
    data['sample'] = ['_'.join(x.split('_')[:2]) for x in samples]
    data['dilution'] = [x.split('_')[2] for x in samples]

    return data


def plot_dilutions(data, metabolites):

    for m in metabolites:
        g = seaborn.FacetGrid(data, col="sample", col_wrap=2, aspect=1.25)
        g.map(seaborn.boxplot, "dilution", m, order=['0001', '0002', '0004', '0008', '0016', '0032', '0064'])
        pyplot.tight_layout()

    pyplot.show()


def assemble_dataset(pp_data, aa_data):

    spp = pp_data.loc[pp_data['sample'] == 'P2_SPP', :].sort_index()
    pp = pp_data.loc[pp_data['sample'] == 'P1_PP', :].sort_index()
    srm = pp_data.loc[pp_data['sample'] == 'P2_SRM', :].sort_index()

    Y = pandas.DataFrame()
    for m in pps:
        Y = pandas.concat([Y.reset_index(drop=True), spp[m].reset_index(drop=True)])

    X = pandas.DataFrame()
    for m in pps:
        df = pandas.concat([
            pp[m].reset_index(drop=True),
            srm[m].reset_index(drop=True),
            srm['dilution'].astype('int').reset_index(drop=True)
        ], axis=1)

        df.columns = ['in_water', 'in_srm', 'dilution']
        df['compound'] = m
        df['compound_class'] = 'PP'
        df['mz'] = mz_dict[m]
        X = pandas.concat([X.reset_index(drop=True), df.reset_index(drop=True)])

    saa = aa_data.loc[aa_data['sample'] == 'P2_SAA', :].sort_index()
    aa = aa_data.loc[aa_data['sample'] == 'P1_AA', :]
    aa.drop(index=('P1_AA_0064_0108_2'), inplace=True)  # this one is missing in P2_SAA
    srm = aa_data.loc[aa_data['sample'] == 'P2_SRM', :]
    srm.drop(index=('P2_SRM_0064_0108_2'), inplace=True)  # this one is missing in P2_SAA

    for m in aas:
        Y = pandas.concat([Y.reset_index(drop=True), saa[m].reset_index(drop=True)])

    for m in aas:
        df = pandas.concat([
            aa[m].reset_index(drop=True),
            srm[m].reset_index(drop=True),
            srm['dilution'].astype('int').reset_index(drop=True)
        ], axis=1)
        df.columns = ['in_water', 'in_srm', 'dilution']
        df['compound'] = m
        df['compound_class'] = 'AA'
        df['mz'] = mz_dict[m]
        X = pandas.concat([X.reset_index(drop=True), df.reset_index(drop=True)])

    full = pandas.concat([Y, X], axis=1)
    full.columns = ['spiked_in', *list(X.columns)]

    return full


if __name__ == '__main__':

    path = '/Users/andreidm/ETH/projects/normalization/res/SRM_v7/6b8943e1/normalized_6b8943e1.csv'
    normalized_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], pps)
    normalized_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], aas)

    dataset = assemble_dataset(normalized_pp, normalized_aa)






