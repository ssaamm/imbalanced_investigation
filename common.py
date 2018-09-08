import json
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

dir = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(dir, 'datasets')
try:
    os.mkdir(DATASETS_DIR)
except FileExistsError:
    pass  # that's ok


def _init_tracker():
    try:
        with open('targets.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_tracker():
    if get_tracker._tracker is None:
        get_tracker._tracker = _init_tracker()
    return get_tracker._tracker

get_tracker._tracker = None


def write_tracker():
    with open('targets.json', 'w') as f:
        json.dump(get_tracker._tracker, f)


def write_dataset(df, dataset_name, target):
    targets = get_tracker()
    targets[dataset_name] = target
    if frozenset(df[target].unique()) == frozenset(('positive', 'negative')):
        df[target] = df[target] == 'positive'
    df.to_csv(os.path.join(DATASETS_DIR, dataset_name), index=False)


def get_logger(name):
    return logging.getLogger(name)
