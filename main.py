import os
import multiprocessing as mp
import itertools as it

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model as lm
import sklearn.metrics as mx
from imblearn.combine import SMOTEENN, SMOTETomek
from tqdm import tqdm

import common

logger = common.get_logger(__name__)

def evaluate_clf(df, train_ndx, test_ndx, features, target, clfs, resamplers, dataset_name):
    X_train = df.loc[train_ndx, features]
    X_test = df.loc[test_ndx, features]
    y_train = df.loc[train_ndx, target]
    y_test = df.loc[test_ndx, target]

    scl = StandardScaler().fit(X_train)
    Xp_train = scl.transform(X_train)
    Xp_test = scl.transform(X_test)

    all_results = []
    for resampler, clf in it.product(resamplers, clfs):
        if resampler is not None:
            try:
                Xp_train, y_train = resampler().fit_sample(Xp_train, y_train)
            except ValueError as x:
                logger.warning(f'Unable to resample with {resampler.__name__} for {dataset_name}')
                continue

        clf.fit(Xp_train, y_train)
        y_pred = clf.predict(Xp_test)

        all_results.append({
            'dataset': dataset_name,
            'clf': clf.__class__.__name__,
            'resampler': resampler.__name__ if resampler else 'None',
            'f1': mx.f1_score(y_test, y_pred),
        })

    return all_results

if __name__ == '__main__':
    targets = common.get_tracker()
    rows = []

    # p = mp.Pool()
    futures = []
    for dataset_name, target in targets.items():
        df = pd.read_csv(os.path.join(common.DATASETS_DIR, dataset_name))
        vc = df[target].value_counts()
        biggest_class_proportion = max(vc / vc.sum())

        logger.info(f'Working with {dataset_name}, shape={df.shape}, largest class={biggest_class_proportion:.0%}')
        features = [c for c in df if c != target]

        skf = StratifiedKFold(n_splits=10)
        resamplers = (None, SMOTEENN, SMOTETomek)
        clfs = (lm.LogisticRegression(solver='lbfgs'),)
        for train, test in skf.split(df[features], df[target]):
            evaluate_clf(df, train, test, features=features, target=target, clfs=clfs, resamplers=resamplers, dataset_name=dataset_name)
            # futures.append(p.apply_async(evaluate_clf, (df, train, test),
            #                              kwds=dict(features=features, target=target,
            #                                        clfs=[lm.LogisticRegression(solver='lbfgs')],
            #                                        resamplers=resamplers,
            #                                        dataset_name=dataset_name,)))
    rows = []
    for f in tqdm(futures):
        rows.append(f.get())

    p.close()

