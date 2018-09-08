import os

from typing import Iterable
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

import keel
import common

logger = common.get_logger(__name__)

def find_datasets(base_folder='datasets') -> Iterable[str]:
    for root, _, files in os.walk(base_folder):
        if any(f.endswith('.zip') for f in files):
            file = next(f for f in files if f.endswith('zip')
                        and not f.endswith('5-fold.zip'))
            yield os.path.join(root, file)

if __name__ == '__main__':
    dtypes = set()

    for fn in find_datasets(os.path.expanduser('~/dev/notebooks/imbalanced_stuff/datasets')):
        logger.info(f'reading {fn}')
        meta, df = keel.read_dataset(fn)

        _, dataset_name = os.path.split(fn)
        dataset_name = dataset_name.replace('zip', 'csv')

        df.replace('<null>', np.nan, inplace=True)

        categorical_columns = [a.name for a in meta['attributes']
                               if a.name in meta['inputs']
                               and a.data_type.startswith('{')]
        if categorical_columns:
            other_features = [a for a in meta['inputs'] if a not in categorical_columns]

            ct = ColumnTransformer(transformers=[('cat', OneHotEncoder(sparse=False), categorical_columns),])
            encoded_features = ct.fit_transform(df[categorical_columns])
            feature_names = ct.named_transformers_['cat'].get_feature_names()

            df = pd.concat([pd.DataFrame(encoded_features, columns=feature_names),
                            df[other_features],
                            df[meta['outputs']]], axis=1)

        null_columns = pd.isnull(df[[c for c in df.columns if c != meta['outputs']]]).sum()
        null_columns = list(null_columns[null_columns > 0].index)
        if null_columns:
            ct = ColumnTransformer(transformers=[('si', SimpleImputer(strategy='median'), null_columns)])
            imputed_features = ct.fit_transform(df[null_columns])
            df[null_columns] = imputed_features

        common.write_dataset(df, dataset_name, target=meta['outputs'])

    common.write_tracker()
