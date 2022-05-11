
import pandas, seaborn, random
from matplotlib import pyplot

if __name__ == '__main__':

    filtered = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/filtered_data_v7.csv', index_col=0).T.sort_index()
    normalized = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/SRM_v7/6b8943e1/normalized_6b8943e1.csv', index_col=0).T.sort_index()

    samples = []
    for name in filtered.index:
        if str(name).split('_')[2] in ['0001', '0002', '0004', '0008', '0016', '0032', '0064'] and str(name).split('_')[1] in ['PP', 'SPP', 'SRM']:
            samples.append(name)

    filtered = filtered.loc[samples, :]
    normalized = normalized.loc[samples, :]

    filtered['sample'] = ['_'.join(x.split('_')[:2]) for x in samples]
    filtered['dilution'] = [x.split('_')[2] for x in samples]
    normalized['sample'] = ['_'.join(x.split('_')[:2]) for x in samples]
    normalized['dilution'] = [x.split('_')[2] for x in samples]

    metabolites = random.sample(list(filtered.columns), 20)

    # for m in metabolites:
    for m in list(filtered.columns):

        g = seaborn.FacetGrid(filtered, col="sample", col_wrap=2, aspect=1.25)
        g.map(seaborn.boxplot, "dilution", m, order=['0001', '0002', '0004', '0008', '0016', '0032', '0064'])
        pyplot.suptitle('filtered')
        pyplot.tight_layout()

        # g = seaborn.FacetGrid(normalized, col="sample", col_wrap=2, aspect=1.25)
        # g.map(seaborn.boxplot, "dilution", m, order=['0001', '0002', '0004', '0008', '0016', '0032', '0064'])
        # pyplot.suptitle('normalized')
        # pyplot.tight_layout()
        pyplot.show()
