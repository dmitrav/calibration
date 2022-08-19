
import numpy, pandas, seaborn, time, os
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.svm import SVR

from constants import aas, pps, mz_dict, parse_batch_label, save_to, split_ratio, seed, n_jobs, get_compounds_classes
from predicting_intensities import get_data


def assemble_dataset(pp_data, aa_data, sample_type='SRM'):

    # spp = pp_data.loc[pp_data['sample'] == 'P2_SPP', :].sort_index()
    water = pp_data.loc[pp_data['sample'] == 'P1_PP', :].sort_index()
    if sample_type == 'SRM':
        serum = pp_data.loc[pp_data['sample'] == 'P2_SRM', :].sort_index()
    elif sample_type == 'spike-in':
        serum = pp_data.loc[pp_data['sample'] == 'P2_SPP', :].sort_index()
    else:
        raise ValueError()

    Y = pandas.DataFrame()
    for m in pps:
        Y = pandas.concat([Y.reset_index(drop=True), serum[m].reset_index(drop=True)])

    X = pandas.DataFrame()
    for m in pps:
        df = pandas.concat([
            # pp[m].reset_index(drop=True),
            water[m].reset_index(drop=True),
            serum['dilution'].astype('int').reset_index(drop=True),
            pandas.Series([parse_batch_label(x) for x in list(serum[m].index)])
        ], axis=1)

        df.columns = ['in_water', 'dilution', 'batch']
        df['compound'] = m
        df['compound_class'] = 'PP'
        df['mz'] = mz_dict[m]
        X = pandas.concat([X.reset_index(drop=True), df.reset_index(drop=True)])

    water = aa_data.loc[aa_data['sample'] == 'P1_AA', :].sort_index()
    if sample_type == 'spike-in':
        water.drop(index=('P1_AA_0064_0108_2'), inplace=True)  # this one is missing in P2_SAA

    if sample_type == 'SRM':
        serum = aa_data.loc[aa_data['sample'] == 'P2_SRM', :].sort_index()
        # sample.drop(index=('P2_SRM_0064_0108_2'), inplace=True)  # this one is missing in P2_SAA
    elif sample_type == 'spike-in':
        serum = aa_data.loc[aa_data['sample'] == 'P2_SAA', :].sort_index()

    for m in aas:
        Y = pandas.concat([Y.reset_index(drop=True), serum[m].reset_index(drop=True)])

    for m in aas:
        df = pandas.concat([
            # aa[m].reset_index(drop=True),
            water[m].reset_index(drop=True),
            serum['dilution'].astype('int').reset_index(drop=True),
            pandas.Series([parse_batch_label(x) for x in list(serum[m].index)])
        ], axis=1)
        # df.columns = ['in_water', 'in_srm', 'dilution']
        df.columns = ['in_water', 'dilution', 'batch']
        df['compound'] = m
        df['compound_class'] = 'AA'
        df['mz'] = mz_dict[m]
        X = pandas.concat([X.reset_index(drop=True), df.reset_index(drop=True)])

    X = X.reset_index(drop=True)
    # calculate relative abundances
    for m in [*pps, *aas]:

        for b in X['batch'].unique():
            for d in X['dilution'].unique():
                if d > 1:
                    Y.iloc[X.loc[(X['compound'] == m) & (X['batch'] == b) & (X['dilution'] == d)].index] \
                        /= Y.iloc[X.loc[(X['compound'] == m) & (X['batch'] == b) & (X['dilution'] == 1)].index].mean()
                else:
                    pass
            # case d == 1 in the end of the loop
            Y.iloc[X.loc[(X['compound'] == m) & (X['batch'] == b) & (X['dilution'] == 1)].index] \
                /= Y.iloc[X.loc[(X['compound'] == m) & (X['batch'] == b) & (X['dilution'] == 1)].index].mean()

    X = pandas.get_dummies(X)
    Y.columns = ['relabu']

    return X, Y


def train_models(X, Y, plot_id='', save_to=save_to):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=split_ratio, random_state=seed)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR())
    ])

    param_grid = {
        'regressor__C': [0.01, 0.1, 1, 10, 100, 1000],
        'regressor__kernel': ['linear', 'rbf', 'sigmoid'],
        'regressor__epsilon': [0, 0.0001, 0.001, 0.01, 0.1]
    }

    scoring = {"r2": make_scorer(r2_score), "mse": make_scorer(mean_squared_error, greater_is_better=False)}
    reg = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, refit='r2', cv=5, n_jobs=n_jobs)

    start = time.time()
    reg.fit(X_train, y_train.values.ravel())
    print('training for took {} min'.format(round(time.time() - start) // 60 + 1))
    print(reg.best_params_)

    predictions = reg.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print('r2 = {:.3f}, mse = {:.3f}'.format(r2, mse))

    data = pandas.DataFrame({'true': y_test.values.reshape(-1), 'predicted': predictions,
                             'batch': X_test['batch'].values.reshape(-1) + 1,
                             'mz': X_test['mz'].values.reshape(-1),
                             'compound_class': get_compounds_classes(X_test)})

    pyplot.figure(figsize=(7, 7))
    seaborn.scatterplot(data=data, x='true', y='predicted', hue='batch', style='compound_class', size='mz',
                        palette='dark')
    pyplot.ylim(0, 3)
    pyplot.xlim(0, 3)
    pyplot.xlabel('relative abundance (true)')
    pyplot.ylabel('relative abundance (predicted)')
    pyplot.title('R2 = {:.3f}, MSE = {:.3f}'.format(r2, mse))

    pyplot.grid()
    pyplot.tight_layout()
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    pyplot.savefig(save_to + 'regression_{}.pdf'.format(plot_id))
    pyplot.close()
    # pyplot.show()

    return reg


if __name__ == '__main__':

    path = '/Users/andreidm/ETH/projects/calibration/data/filtered_data.csv'
    initial_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], metabolites=pps)
    initial_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], metabolites=aas)
    X, Y = assemble_dataset(initial_pp, initial_aa, sample_type='SRM')

    print('training for initial data\n')
    best_svr = train_models(X, Y, plot_id='initial', save_to=save_to + 'relabu/with_water/')

    X_test, Y_test = assemble_dataset(initial_pp, initial_aa, sample_type='spike-in')

    predictions = best_svr.predict(X_test)
    r2 = r2_score(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    print('r2 = {:.3f}, mse = {:.3f}'.format(r2, mse))

    data = pandas.DataFrame({'true': Y_test.values.reshape(-1), 'predicted': predictions,
                             'batch': X_test['batch'].values.reshape(-1) + 1,
                             'mz': X_test['mz'].values.reshape(-1),
                             'compound_class': get_compounds_classes(X_test)})

    pyplot.figure(figsize=(7, 7))
    seaborn.scatterplot(data=data, x='true', y='predicted', hue='batch', style='compound_class', size='mz',
                        palette='dark')
    pyplot.xlabel('relative abundance (true)')
    pyplot.ylabel('relative abundance (predicted)')
    pyplot.title('Testing on SRM + spike-ins: R2 = {:.3f}, MSE = {:.3f}'.format(r2, mse))

    pyplot.grid()
    pyplot.tight_layout()
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    pyplot.savefig(save_to + 'relabu/with_water/regression_testing.pdf')
    pyplot.close()
    # pyplot.show()


