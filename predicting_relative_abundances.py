
import numpy, pandas, seaborn, time, os, itertools
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.svm import SVR

from constants import aas, pps, mz_dict, parse_batch_label, save_to
from constants import split_ratio, seed, n_jobs, get_compounds_classes, dilutions
from predicting_intensities import get_data


def assemble_dataset(pp_data, aa_data, sample_type='SRM', dils=None):

    if sample_type == 'SRM':
        serum = pp_data.loc[pp_data['sample'] == 'P2_SRM', :].sort_index()
        if dils:
            serum = serum.loc[serum['dilution'].isin(dils)]
    elif sample_type == 'spike-in':
        serum = pp_data.loc[pp_data['sample'] == 'P2_SPP', :].sort_index()
        if dils:
            serum = serum.loc[serum['dilution'].isin(dils)]
    else:
        raise ValueError()

    Y = pandas.DataFrame()
    for m in pps:
        Y = pandas.concat([Y.reset_index(drop=True), serum[m].reset_index(drop=True)])

    X = pandas.DataFrame()
    for m in pps:
        df = pandas.concat([
            serum['dilution'].astype('int').reset_index(drop=True),
            pandas.Series([parse_batch_label(x) for x in list(serum[m].index)])
        ], axis=1)
        df.columns = ['dilution', 'batch']
        df['compound'] = m
        df['compound_class'] = 'PP'
        df['mz'] = mz_dict[m]
        X = pandas.concat([X.reset_index(drop=True), df.reset_index(drop=True)])

    if sample_type == 'SRM':
        serum = aa_data.loc[aa_data['sample'] == 'P2_SRM', :].sort_index()
        if dils:
            serum = serum.loc[serum['dilution'].isin(dils)]
        # sample.drop(index=('P2_SRM_0064_0108_2'), inplace=True)  # this one is missing in P2_SAA
    elif sample_type == 'spike-in':
        serum = aa_data.loc[aa_data['sample'] == 'P2_SAA', :].sort_index()
        if dils:
            serum = serum.loc[serum['dilution'].isin(dils)]

    for m in aas:
        Y = pandas.concat([Y.reset_index(drop=True), serum[m].reset_index(drop=True)])

    for m in aas:
        df = pandas.concat([
            serum['dilution'].astype('int').reset_index(drop=True),
            pandas.Series([parse_batch_label(x) for x in list(serum[m].index)])
        ], axis=1)
        df.columns = ['dilution', 'batch']
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
    Y = Y.reset_index(drop=True)
    Y.columns = ['relabu']

    return X, Y


def train_models(X, Y, plot_id='', save_to=save_to):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=split_ratio, random_state=seed)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR())
    ])

    param_grid = {
        'regressor__C': [0.01, 0.1, 1, 10, 100],
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

    if 'spike_in' in plot_id:
        pyplot.ylim(0, 1.5)
        pyplot.xlim(0, 1.5)
    else:
        pyplot.ylim(0, 3)
        pyplot.xlim(0, 3)
    pyplot.xlabel('relative abundance (true)')
    pyplot.ylabel('relative abundance (predicted)')
    pyplot.title('R2 = {:.3f}, MSE = {:.3f}'.format(r2, mse))

    pyplot.grid()
    pyplot.tight_layout()
    # if not os.path.exists(save_to):
    #     os.makedirs(save_to)
    # pyplot.savefig(save_to + 'regression_{}.pdf'.format(plot_id))
    # pyplot.close()
    pyplot.show()

    return reg, r2, mse


def train_on_srm_and_test_on_spikeins():

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


def train_and_test_with_spike_ins_only():

    path = '/Users/andreidm/ETH/projects/calibration/data/filtered_data.csv'
    initial_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], metabolites=pps)
    initial_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], metabolites=aas)
    X, Y = assemble_dataset(initial_pp, initial_aa, sample_type='spike-in')

    print('training for initial data\n')
    best_svr = train_models(X, Y, plot_id='initial_spike_ins', save_to=save_to + 'relabu/')


def train_with_supplemented_spike_ins():

    path = '/Users/andreidm/ETH/projects/calibration/data/filtered_data.csv'
    initial_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], metabolites=pps)
    initial_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], metabolites=aas)

    X, Y = assemble_dataset(initial_pp, initial_aa, sample_type='SRM')

    # supplement combinations of dilutions
    for k in range(2, len(dilutions)):
        # pick randomly k dilutions to train on
        dilutions_train = list(itertools.combinations(dilutions, k))

        for combination in dilutions_train:
            if '0001' not in combination:
                pass  # always keep an undiluted sample
            else:
                eXtra, eYtra = assemble_dataset(initial_pp, initial_aa, sample_type='spike-in', dils=combination)
                newX = pandas.concat([X.reset_index(drop=True), eXtra.reset_index(drop=True)]).reset_index(drop=True)
                newY = pandas.concat([Y.reset_index(drop=True), eYtra.reset_index(drop=True)]).reset_index(drop=True)
                print('training for supplemented combination: {}'.format(combination))
                best_svr, r2, mse = train_models(newX, newY, plot_id='_'.join(combination),
                                                 save_to=save_to + 'relabu/training_with_spike_ins/')


def test_best_models_with_supplemented_spike_ins():

    path = '/Users/andreidm/ETH/projects/calibration/data/filtered_data.csv'
    initial_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], metabolites=pps)
    initial_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], metabolites=aas)

    X, Y = assemble_dataset(initial_pp, initial_aa, sample_type='SRM')

    for combination in [('0001', '0002'), ('0001', '0002', '0004')]:  # best r2, best mse
        eXtra, eYtra = assemble_dataset(initial_pp, initial_aa, sample_type='spike-in', dils=combination)
        newX = pandas.concat([X.reset_index(drop=True), eXtra.reset_index(drop=True)]).reset_index(drop=True)
        newY = pandas.concat([Y.reset_index(drop=True), eYtra.reset_index(drop=True)]).reset_index(drop=True)
        print('training for supplemented combination: {}'.format(combination))
        best_svr, _, _ = train_models(newX, newY, plot_id='_'.join(combination),
                                      save_to=save_to + 'relabu/training_with_spike_ins/')

        X_test, Y_test = assemble_dataset(initial_pp, initial_aa, sample_type='spike-in',
                                          dils=[x for x in dilutions if x not in combination or x == '0001'])

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
        pyplot.savefig(
            save_to + 'relabu/training_with_spike_ins/regression_testing_{}.pdf'.format('_'.join(combination)))
        pyplot.close()
        # pyplot.show()


def compare_calibration_with_ralps():

    path = '/Users/andreidm/ETH/projects/calibration/data/filtered_data.csv'
    initial_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], metabolites=pps)
    initial_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], metabolites=aas)
    X, Y = assemble_dataset(initial_pp, initial_aa, sample_type='spike-in')

    print('training for initial data\n')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=split_ratio, random_state=seed)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR())
    ])

    param_grid = {
        'regressor__C': [0.01, 0.1, 1, 10, 100],
        'regressor__kernel': ['linear', 'rbf', 'sigmoid'],
        'regressor__epsilon': [0, 0.0001, 0.001, 0.01, 0.1]
    }

    scoring = {"r2": make_scorer(r2_score), "mse": make_scorer(mean_squared_error, greater_is_better=False)}
    reg = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, refit='r2', cv=5, n_jobs=n_jobs)

    start = time.time()
    reg.fit(X_train, y_train.values.ravel())
    print('training for took {} min'.format(round(time.time() - start) // 60 + 1))
    print(reg.best_params_)

    predictions = pandas.DataFrame(reg.predict(X_test), index=y_test.index)

    results = {'batch': [], 'cv': [], 'method': []}
    for batch in X_test['batch'].unique():
        results['batch'].append(batch + 1)
        values = predictions.loc[X_test.loc[X_test['batch'] == batch].index].values
        results['cv'].append(numpy.std(values) / numpy.mean(values))
        results['method'].append('calibration')

    # get RALPS results
    path = '/Users/andreidm/ETH/projects/calibration/data/SRM_SPP_normalized_2b632f6b.csv'
    normalized_pp = get_data(path, ['P1_PP', 'P2_SPP', 'P2_SRM'], metabolites=pps)
    normalized_aa = get_data(path, ['P1_AA', 'P2_SAA', 'P2_SRM'], metabolites=aas)
    _, Y = assemble_dataset(initial_pp, initial_aa, sample_type='spike-in')

    for batch in X_test['batch'].unique():
        results['batch'].append(batch + 1)
        values = Y.loc[X_test.loc[X_test['batch'] == batch].index].values
        results['cv'].append(numpy.std(values) / numpy.mean(values))
        results['method'].append('RALPS')

    results = pandas.DataFrame(results)
    seaborn.barplot(x='batch', y='cv', hue='method', data=results)
    pyplot.title('Variation coefficient per batch')
    pyplot.tight_layout()
    pyplot.savefig(save_to + 'relabu/comparison.pdf')


if __name__ == '__main__':

    train_on_srm_and_test_on_spikeins()
