

import os.path

import pandas, seaborn, random, time, numpy, seaborn, itertools
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.svm import SVR
from matplotlib import pyplot
from tqdm import tqdm

from constants import dilutions, aas, pps, mz_dict
from constants import seed, split_ratio
from constants import save_to, n_jobs, parse_batch_label
from constants import normalized_pp_metabolites, normalized_aa_metabolites, initial_aa_outliers, initial_pp_outliers


def get_data(path, sample_codes, metabolites=None):

    data = pandas.read_csv(path, index_col=0).T.sort_index()
    data[data < 0] = 1

    samples = []
    for name in data.index:
        if str(name).split('_')[2] in dilutions and '_'.join(str(name).split('_')[:2]) in sample_codes:
            samples.append(name)

    if metabolites:
        data = data.loc[samples, metabolites]
    else:
        data = data.loc[samples, :]

    data['sample'] = ['_'.join(x.split('_')[:2]) for x in samples]
    data['dilution'] = [x.split('_')[2] for x in samples]

    return data


def assemble_dataset(pp_data, aa_data):

    spp = pp_data.loc[pp_data['sample'] == 'P2_SPP', :].sort_index()
    pp = pp_data.loc[pp_data['sample'] == 'P1_PP', :].sort_index()

    Y = pandas.DataFrame()
    for m in pps:
        Y = pandas.concat([Y.reset_index(drop=True), spp['dilution'].astype('int').reset_index(drop=True)])

    X = pandas.DataFrame()
    for m in pps:
        df = pandas.concat([
            pp[m].reset_index(drop=True),
            spp[m].reset_index(drop=True),
            pandas.Series([parse_batch_label(x) for x in list(spp[m].index)])
        ], axis=1)

        df.columns = ['water_spike_in', 'srm_spike_in', 'batch']
        df['compound'] = m
        df['compound_class'] = 'PP'
        df['mz'] = mz_dict[m]
        X = pandas.concat([X.reset_index(drop=True), df.reset_index(drop=True)])

    saa = aa_data.loc[aa_data['sample'] == 'P2_SAA', :].sort_index()
    aa = aa_data.loc[aa_data['sample'] == 'P1_AA', :].sort_index()
    aa.drop(index=('P1_AA_0064_0108_2'), inplace=True)  # this one is missing in P2_SAA

    for m in aas:
        Y = pandas.concat([Y.reset_index(drop=True), saa['dilution'].astype('int').reset_index(drop=True)])

    for m in aas:
        df = pandas.concat([
            aa[m].reset_index(drop=True),
            saa[m].reset_index(drop=True),
            pandas.Series([parse_batch_label(x) for x in list(saa[m].index)])
        ], axis=1)
        df.columns = ['water_spike_in', 'srm_spike_in', 'batch']
        df['compound'] = m
        df['compound_class'] = 'AA'
        df['mz'] = mz_dict[m]
        X = pandas.concat([X.reset_index(drop=True), df.reset_index(drop=True)])

    X = pandas.get_dummies(X)
    Y.columns = ['dilution']

    return X, Y


def remove_outliers(X, Y, threshold, column='srm_spike_in'):
    data = pandas.concat([Y, X], axis=1)
    data = data[data[column] < threshold]
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


def train_with_full_data(X, Y,
                         outlier_thresholds=[2e6, 1e7],  # (best scores, all data points) for RALPS
                         split_ratio=split_ratio,
                         random_seed=seed,
                         save_to=save_to,
                         plot_id=''):

    for t in outlier_thresholds:

        fX, fY = remove_outliers(X, Y, t)  # removes a long tail of single outliers
        print('shape with t = {}: {}'.format(t, fY.shape))
        X_train, X_test, y_train, y_test = train_test_split(fX, fY, train_size=split_ratio, random_state=random_seed)

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
        reg.fit(X_train, numpy.log2(y_train).values.ravel())
        print('training for took {} min'.format(round(time.time() - start) // 60 + 1))
        print(reg.best_params_)

        predictions = reg.predict(X_test)
        r2 = r2_score(numpy.log2(y_test).values.ravel(), predictions)
        mse = mean_squared_error(numpy.log2(y_test).values.ravel(), predictions)
        print('r2 = {:.3f}, mse = {:.3f}'.format(r2, mse))

        data = pandas.DataFrame({'true': numpy.log2(y_test).values.ravel(), 'predicted': predictions,
                                 'batch': X_test['batch'].values.reshape(-1) + 1,
                                 'mz': X_test['mz'].values.reshape(-1),
                                 'compound_class': get_compounds_classes(X_test)})

        pyplot.figure(figsize=(7,7))
        seaborn.scatterplot(data=data, x='true', y='predicted', hue='batch', style='compound_class', size='mz', palette='dark')
        pyplot.ylim(-1,7)
        pyplot.xlabel('log2 dilution (true)')
        pyplot.ylabel('log2 dilution (predicted)')
        if t >= 1e7:
            pyplot.title('R2 = {:.3f}, MSE = {:.3f}, all values'.format(r2, mse))
        else:
            pyplot.title('R2 = {:.3f}, MSE = {:.3f}, values < {:.1e}'.format(r2, mse, t))
        pyplot.grid()
        pyplot.tight_layout()

        if not os.path.exists(save_to):
            os.makedirs(save_to)
        pyplot.savefig(save_to + 'regression_t={:.1e}_{}.pdf'.format(t, plot_id))
        pyplot.close()
    # pyplot.show()


def train_with_water_only(X, Y,
                         outlier_thresholds=[4.4e7],  # (best scores, all data points) for RALPS
                         save_to=save_to,
                         plot_id=''):

    for t in outlier_thresholds:

        fX, fY = remove_outliers(X, Y, t, column='water_spike_in')  # removes a long tail of single outliers
        print('shape with t = {}: {}'.format(t, fY.shape))

        srm_spike_in = fX['srm_spike_in']
        fX = fX.drop(columns=['srm_spike_in'])

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(max_iter=5000))
        ])

        param_grid = {
            'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'regressor__l1_ratio': [0., 0.2, 0.4, 0.6, 0.8, 1],
            'regressor__fit_intercept': [True, False]
        }

        scoring = {"r2": make_scorer(r2_score), "mse": make_scorer(mean_squared_error, greater_is_better=False)}
        reg = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, refit='r2', cv=5, n_jobs=-1)

        start = time.time()
        reg.fit(fX, numpy.log2(fY).values.ravel())
        print('training for took {} min'.format(round(time.time() - start) // 60 + 1))
        print(reg.best_params_)

        X_test = fX.copy()
        # substitute ion counts in water with rescaled ion counts in SRM and predict
        X_test['water_spike_in'] = srm_spike_in
        predictions = reg.predict(X_test)

        r2 = r2_score(numpy.log2(fY).values.ravel(), predictions)
        mse = mean_squared_error(numpy.log2(fY).values.ravel(), predictions)
        print('r2 = {:.3f}, mse = {:.3f}'.format(r2, mse))

        data = pandas.DataFrame({'true': numpy.log2(fY).values.ravel(), 'predicted': predictions,
                                 'batch': X_test['batch'].values.reshape(-1) + 1,
                                 'mz': X_test['mz'].values.reshape(-1),
                                 'compound_class': get_compounds_classes(X_test)})

        pyplot.figure(figsize=(7,7))
        seaborn.scatterplot(data=data, x='true', y='predicted', hue='batch', style='compound_class', size='mz', palette='dark')
        pyplot.xlabel('log2 dilution (true)')
        pyplot.ylabel('log2 dilution (predicted)')
        if t == 4.4e7:
            pyplot.title('R2 = {:.3f}, MSE = {:.3f}, all values'.format(r2, mse))
        else:
            pyplot.title('R2 = {:.3f}, MSE = {:.3f}, values < {:.1e}'.format(r2, mse, t))
        pyplot.grid()
        pyplot.tight_layout()

        # if not os.path.exists(save_to):
        #     os.makedirs(save_to)
        # pyplot.savefig(save_to + 'regression_t={:.1e}_{}.pdf'.format(t, plot_id))
        # pyplot.close()
    pyplot.show()


def train_with_missing_dilutions(X, Y, save_to=save_to, plot_id=''):

    data = pandas.concat([Y, X], axis=1)

    results = {'n_dilutions': [], 'score': [], 'metric': []}
    for k in range(2, len(dilutions)):
        # pick randomly k dilutions to train on
        dilutions_train = list(itertools.combinations(dilutions, k))

        scores = []
        for combination in dilutions_train:

            combination = [int(x) for x in combination]

            # if combination not in [[1, 16, 32], [1, 32, 64], [2, 32, 64], [1, 4, 8, 16, 32, 64]]:  # initial
            # if combination not in [[8, 32, 64]]:  # RALPS
            # if combination not in [[1, 4, 8], [1, 4, 32], [2, 32, 64]]:  # WaveICA
            if combination not in [[2, 16, 32], [2, 16, 64], [1, 4, 16, 64], [2, 8, 16, 32, 64]]:  # ComBat

                df = data.loc[data['dilution'].isin(combination), :]
                X_train = df.iloc[:, 1:]
                y_train = df.iloc[:, 0]

                df = data.loc[~data['dilution'].isin(combination), :]
                X_test = df.iloc[:, 1:]
                y_test = df.iloc[:, 0]

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', SVR())
                ])

                param_grid = {
                    'regressor__C': [0.01, 0.1, 1, 10, 100, 1000],
                    'regressor__kernel': ['linear', 'rbf', 'sigmoid'],
                    'regressor__epsilon': [0, 0.0001, 0.001, 0.01, 0.1]
                }

                reg = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='r2', cv=3, n_jobs=n_jobs)

                # print('for combination {}'.format(combination))
                reg.fit(X_train, numpy.log2(y_train).values.ravel())

                predictions = reg.predict(X_test)
                r2 = r2_score(numpy.log2(y_test).values.ravel(), predictions)
                mse = mean_squared_error(numpy.log2(y_test).values.ravel(), predictions)
                scores.append((r2, mse, reg.best_params_, combination))

        print('best scores for {} dilutions:'.format(k))
        best = sorted(scores, key=lambda x: x[0])[-1]
        results['n_dilutions'].append(k)
        results['score'].append(best[0])
        results['metric'].append('R2')
        print('{} -> {} -> r2 = {:.3f}'.format(best[3], best[2], best[0]))

        best = sorted(scores, key=lambda x: x[1])[0]
        results['n_dilutions'].append(k)
        results['score'].append(best[1])
        results['metric'].append('MSE')
        print('{} -> {} -> mse = {:.3f}'.format(best[3], best[2], best[1]))

    df = pandas.DataFrame(results)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    seaborn.set_theme(style='whitegrid')
    seaborn.barplot(x='n_dilutions', y='score', data=df[df['metric'] == 'R2'])
    pyplot.title('Best R2 scores')
    pyplot.savefig(save_to + 'missing_dilutions_r2_{}.pdf'.format(plot_id))
    pyplot.close()

    seaborn.barplot(x='n_dilutions', y='score', data=df[df['metric'] == 'MSE'])
    pyplot.title('Best MSE scores')
    pyplot.savefig(save_to + 'missing_dilutions_mse_{}.pdf'.format(plot_id))
    pyplot.close()


def train_with_missing_metabolites(X, Y, metabolite_group='aas', plot_id='', save_to=save_to):

    data = pandas.concat([Y, X], axis=1)

    if metabolite_group == 'aas':
        metabolites = aas
        data = data[data['compound_class_AA'] == 1]
    elif metabolite_group == 'pps':
        metabolites = pps
        data = data[data['compound_class_PP'] == 1]
    else:
        raise NotImplementedError('Unknown metabolite group')
    print('for metabolite group: {}'.format(metabolite_group))

    results = {'n_metabolites': [], 'score': [], 'metric': []}
    for k in range(2, len(metabolites)):
        # pick randomly k dilutions to train on
        metabolites_train = list(itertools.combinations(metabolites, k))

        scores = []
        for combination in metabolites_train:

            df = pandas.DataFrame()
            for m in combination:
                df = pandas.concat([df.reset_index(drop=True), data[data['compound_{}'.format(m)] == 1].reset_index(drop=True)])
            X_train = df.iloc[:, 1:]
            y_train = df.iloc[:, 0]

            df = pandas.DataFrame()
            for m in metabolites:
                if m not in combination:
                    df = pandas.concat([df.reset_index(drop=True), data[data['compound_{}'.format(m)] == 1].reset_index(drop=True)])
            X_test = df.iloc[:, 1:]
            y_test = df.iloc[:, 0]

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', SVR())
            ])

            param_grid = {
                'regressor__C': [0.01, 0.1, 1, 10, 100, 1000],
                'regressor__kernel': ['linear', 'rbf', 'sigmoid'],
                'regressor__epsilon': [0, 0.0001, 0.001, 0.01, 0.1]
            }

            reg = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='r2', cv=3, n_jobs=n_jobs)
            reg.fit(X_train, numpy.log2(y_train).values.ravel())

            predictions = reg.predict(X_test)
            r2 = r2_score(numpy.log2(y_test).values.ravel(), predictions)
            mse = mean_squared_error(numpy.log2(y_test).values.ravel(), predictions)
            scores.append((r2, mse, reg.best_params_, combination))

        print('best scores for {} metabolites:'.format(k))
        best = sorted(scores, key=lambda x: x[0])[-1]
        results['n_metabolites'].append(k)
        results['score'].append(best[0])
        results['metric'].append('R2')
        print('{} -> {} -> r2 = {:.3f}'.format(best[3], best[2], best[0]))

        best = sorted(scores, key=lambda x: x[1])[0]
        results['n_metabolites'].append(k)
        results['score'].append(best[1])
        results['metric'].append('MSE')
        print('{} -> {} -> mse = {:.3f}'.format(best[3], best[2], best[1]))

    df = pandas.DataFrame(results)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    seaborn.set_theme(style='whitegrid')
    seaborn.barplot(x='n_metabolites', y='score', data=df[df['metric'] == 'R2'])
    pyplot.title('Best R2 scores')
    pyplot.savefig(save_to + 'missing_{}_r2_{}.pdf'.format(metabolite_group, plot_id))
    pyplot.close()

    seaborn.barplot(x='n_metabolites', y='score', data=df[df['metric'] == 'MSE'])
    pyplot.title('Best MSE scores')
    pyplot.savefig(save_to + 'missing_{}_mse_{}.pdf'.format(metabolite_group, plot_id))
    pyplot.close()


def train_all_models():
    # INITIAL DATA
    path = '/Users/andreidm/ETH/projects/calibration/data/filtered_data.csv'
    initial_pp = get_data(path, ['P1_PP', 'P2_SPP'], metabolites=pps)
    initial_aa = get_data(path, ['P1_AA', 'P2_SAA'], metabolites=aas)
    X, Y = assemble_dataset(initial_pp, initial_aa)
    print('training for initial data\n')
    train_with_full_data(X, Y, outlier_thresholds=[1e7], plot_id='initial', save_to=save_to + 'full/')
    print('training for missing dilutions\n')
    train_with_missing_dilutions(X, Y, plot_id='_initial', save_to=save_to + 'full_missing_dilutions/')
    print('training for missing metabolites\n')
    train_with_missing_metabolites(X, Y, metabolite_group='aas', plot_id='_initial', save_to=save_to + 'full_missing_metabolites/')
    train_with_missing_metabolites(X, Y, metabolite_group='pps', plot_id='_initial', save_to=save_to + 'full_missing_metabolites/')
    # this is clearly not working
    train_with_water_only(X, Y, outlier_thresholds=[4.4e7], plot_id='_initial', save_to=save_to + 'water_only/')

    # RALPS
    path = '/Users/andreidm/ETH/projects/calibration/data/SRM_SPP_normalized_2b632f6b.csv'
    normalized_pp = get_data(path, ['P1_PP', 'P2_SPP'], metabolites=pps)
    normalized_aa = get_data(path, ['P1_AA', 'P2_SAA'], metabolites=aas)
    X, Y = assemble_dataset(normalized_pp, normalized_aa)
    print('training for the normalized data (RALPS)\n')
    train_with_full_data(X, Y, outlier_thresholds=[1e7], plot_id='RALPS', save_to=save_to + 'full/')
    print('training for missing dilutions\n')
    train_with_missing_dilutions(X, Y, plot_id='RALPS', save_to=save_to + 'full_missing_dilutions/')
    print('training for missing metabolites\n')
    train_with_missing_metabolites(X, Y, metabolite_group='aas', plot_id='RALPS', save_to=save_to + 'full_missing_metabolites/')
    train_with_missing_metabolites(X, Y, metabolite_group='pps', plot_id='RALPS', save_to=save_to + 'full_missing_metabolites/')

    # WAVEICA
    path = '/Users/andreidm/ETH/projects/calibration/data/SRM_SPP_other_methods/waveICA1.csv'
    normalized_pp = get_data(path, ['P1_PP', 'P2_SPP'], metabolites=pps)
    normalized_aa = get_data(path, ['P1_AA', 'P2_SAA'], metabolites=aas)
    X, Y = assemble_dataset(normalized_pp, normalized_aa)
    print('training for the normalized data (WaveICA)\n')
    train_with_full_data(X, Y, outlier_thresholds=[1.2e7], plot_id='WaveICA', save_to=save_to + 'full/')
    print('training for missing dilutions\n')
    train_with_missing_dilutions(X, Y, plot_id='WaveICA', save_to=save_to + 'full_missing_dilutions/')
    print('training for missing metabolites\n')
    train_with_missing_metabolites(X, Y, metabolite_group='aas', plot_id='WaveICA', save_to=save_to + 'full_missing_metabolites/')
    train_with_missing_metabolites(X, Y, metabolite_group='pps', plot_id='WaveICA', save_to=save_to + 'full_missing_metabolites/')

    # COMBAT
    path = '/Users/andreidm/ETH/projects/calibration/data/SRM_SPP_other_methods/combat1.csv'
    normalized_pp = get_data(path, ['P1_PP', 'P2_SPP'], metabolites=pps)
    normalized_aa = get_data(path, ['P1_AA', 'P2_SAA'], metabolites=aas)
    X, Y = assemble_dataset(normalized_pp, normalized_aa)
    print('training for the normalized data (ComBat)\n')
    train_with_full_data(X, Y, outlier_thresholds=[1.2e7], plot_id='ComBat', save_to=save_to + 'full/')
    print('training for missing dilutions\n')
    train_with_missing_dilutions(X, Y, plot_id='ComBat', save_to=save_to + 'full_missing_dilutions/')
    print('training for missing metabolites\n')
    train_with_missing_metabolites(X, Y, metabolite_group='aas', plot_id='ComBat', save_to=save_to + 'full_missing_metabolites/')
    train_with_missing_metabolites(X, Y, metabolite_group='pps', plot_id='ComBat', save_to=save_to + 'full_missing_metabolites/')


def plot_dilutions(data, metabolites=None, plot_name='', save_to='/Users/andreidm/ETH/projects/calibration/res/'):

    if not metabolites:
        metabolites = [str(x) for x in data.columns[:-2]]

    data['dilution'] = data['dilution'].astype('int').astype('str') + 'x'
    for m in metabolites:
        print(m)
        g = seaborn.FacetGrid(data, col="sample", col_wrap=2, sharey=False, aspect=1.25)
        g.map(seaborn.boxplot, "dilution", m, order=['1x', '2x', '4x', '8x', '16x', '32x', '64x'])
        pyplot.tight_layout()

        if not os.path.exists(save_to):
            os.makedirs(save_to)
        pyplot.savefig(save_to + '{}_dilutions_{}.pdf'.format(m, plot_name))
        # pyplot.show()


if __name__ == '__main__':

    save_to = '/Users/andreidm/ETH/projects/calibration/res/predicting_dilutions/'

    path = '/Users/andreidm/ETH/projects/calibration/data/filtered_data.csv'
    initial_pp = get_data(path, ['P1_PP', 'P2_SPP'], metabolites=pps)
    initial_aa = get_data(path, ['P1_AA', 'P2_SAA'], metabolites=aas)

    plot_dilutions(initial_aa, aas, plot_name='AA_initial', save_to=save_to + 'dilutions/')
    plot_dilutions(initial_pp, pps, plot_name='PP_initial', save_to=save_to + 'dilutions/')
