import itertools as it
import multiprocessing as mp
import os

import pandas as pd
import sklearn.metrics as mx
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling.prototype_selection import RandomUnderSampler, TomekLinks, OneSidedSelection
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
        # y_score = clf.predict_proba(Xp_test)[:, 0]

        all_results.append({
            'dataset': dataset_name[:-len('.csv')],
            'clf': clf.__class__.__name__,
            'resampler': resampler.__name__ if resampler else 'None',
            'f1': mx.f1_score(y_test, y_pred),
            # 'auc': mx.roc_auc_score(y_test, y_score),
        })

    return all_results

if __name__ == '__main__':
    targets = common.get_tracker()
    rows = []

    p = mp.Pool(mp.cpu_count() - 1)
    futures = []
    for dataset_name, target in targets.items():
        df = pd.read_csv(os.path.join(common.DATASETS_DIR, dataset_name))
        vc = df[target].value_counts()
        biggest_class_proportion = max(vc / vc.sum())

        logger.info(f'Working with {dataset_name}, shape={df.shape}, largest class={biggest_class_proportion:.0%}')
        features = [c for c in df if c != target]

        skf = StratifiedKFold(n_splits=10)
        resamplers = (None, SMOTEENN, SMOTETomek, SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler, TomekLinks, OneSidedSelection)
        clfs = (
            lm.LogisticRegression(solver='lbfgs'),
            lm.RidgeClassifier(),
            GradientBoostingClassifier(),
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            GaussianNB(),
        )
        for train, test in skf.split(df[features], df[target]):
            futures.append(p.apply_async(evaluate_clf, (df, train, test),
                                         kwds=dict(features=features, target=target,
                                                   clfs=clfs,
                                                   resamplers=resamplers,
                                                   dataset_name=dataset_name,)))
    rows = []
    for f in tqdm(futures):
        rows.extend(f.get())
    p.close()

    pd.DataFrame(rows).to_csv('scores.csv', index=False)
