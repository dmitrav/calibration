
import pandas, seaborn, random, time, numpy, seaborn
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.svm import SVR
from matplotlib import pyplot

from constants import dilutions, aas, pps, mz_dict
from constants import seed, split_ratio


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
    aa = aa_data.loc[aa_data['sample'] == 'P1_AA', :].sort_index()
    aa.drop(index=('P1_AA_0064_0108_2'), inplace=True)  # this one is missing in P2_SAA
    srm = aa_data.loc[aa_data['sample'] == 'P2_SRM', :].sort_index()
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

    X = pandas.get_dummies(X)
    Y.columns = ['spiked_in']

    return X, Y


def remove_outliers(X, Y, threshold):
    data = pandas.concat([Y, X], axis=1)
    data = data[data['spiked_in'] < threshold]
    return data.iloc[:, 1:], data.iloc[:, 0]


def get_compounds_classes(features):

    compounds_classes = []
    # hardcoded for two classes in the data only
    for c in list(features['compound_class_AA']):
        if int(c) == 1:
            compounds_classes.append('AA')
        elif int(c) == 0:
            compounds_classes.append('PP')
        else:
            raise NotImplementedError

    return compounds_classes


def train_baseline_model(X, Y, outlier_thresholds, split_ratio=split_ratio, random_seed=seed):

    for t in outlier_thresholds:

        fX, fY = remove_outliers(X, Y, t)  # removes a long tail of single outliers
        print('shape with t = {}: {}'.format(t, fY.shape))
        X_train, X_test, y_train, y_test = train_test_split(fX, fY, train_size=split_ratio, random_state=random_seed)

        # TODO:
        #  - try other models,
        #  - implement systematic hyperparameter search
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(kernel='rbf'))
        ])

        param_grid = {
            'regressor__C': [50, 100, 150],
            'regressor__epsilon': [0, 0.0001, 0.001, 0.01, 0.1]
        }

        scoring = {"r2": make_scorer(r2_score), "mse": make_scorer(mean_squared_error, greater_is_better=False)}
        reg = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, refit='r2', cv=5, n_jobs=-1)

        start = time.time()
        reg.fit(X_train, numpy.log(y_train).values.ravel())
        print('training for took {} min'.format(round(time.time() - start) // 60 + 1))
        print(reg.best_params_)
        results = pandas.DataFrame(reg.cv_results_)

        # TODO:
        #  - look into feature importances
        predictions = numpy.exp(reg.predict(X_test))
        data = pandas.DataFrame({'true': y_test.values.reshape(-1), 'predicted': predictions,
                                 'dilution': X_test['dilution'].values.reshape(-1),
                                 'mz': X_test['mz'].values.reshape(-1),
                                 'compound_class': get_compounds_classes(X_test)})

        pyplot.figure(figsize=(7,7))
        seaborn.scatterplot(data=data, x='true', y='predicted', hue='dilution', style='compound_class', size='mz', palette='muted')
        pyplot.title('R2 = {:.3f}, MSE = {:.3f}, values < {:.1e}'.format(results.loc[reg.best_index_, 'mean_test_r2'],
                                                                     -results.loc[reg.best_index_, 'mean_test_mse'], t))
        pyplot.grid()
        pyplot.tight_layout()
    pyplot.show()


if __name__ == '__main__':

    path = '/Users/andreidm/ETH/projects/normalization/res/SRM_v7/6b8943e1/normalized_6b8943e1.csv'
    normalized_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], pps)
    normalized_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], aas)

    X, Y = assemble_dataset(normalized_pp, normalized_aa)

    thresholds = [0.4e6, 1.1e6, 1e7]  # -> [best MSE, tradeoff, best R2]
    train_baseline_model(X, Y, thresholds)








