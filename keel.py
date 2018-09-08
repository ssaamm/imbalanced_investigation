from typing import Tuple, Dict, BinaryIO
from enum import Enum
from collections import namedtuple
import zipfile as zf

import pandas as pd


State = Enum('State', 'RELATION ATTRIBUTE INPUTS OUTPUTS DATA DATA_TAG_READ')
Attribute = namedtuple('Attribute', 'name data_type other')


def _build_attribute(split):
    if split[0] != '@attribute':
        raise ValueError('must be an attribute')
    return Attribute(name=split[1], data_type=split[2], other=' '.join(split[3:]))


def _process_line(line, state, meta):
    split = line.split()
    if state == State.RELATION:
        meta['relation'] = split[1]
        return State.ATTRIBUTE, meta

    if state == State.ATTRIBUTE:
        if split[0] != '@attribute':
            return _process_line(line, State.INPUTS, meta)

        meta['attributes'].append(_build_attribute(split))
        return State.ATTRIBUTE, meta

    if state == State.INPUTS:
        meta['inputs'] = line[len('@inputs '):].split(', ')
        return State.OUTPUTS, meta

    if state == State.OUTPUTS:
        meta['outputs'] = split[1]
        return State.DATA, meta

    if state == State.DATA:
        if split[0] != '@data':
            raise ValueError('expected data')
        return State.DATA_TAG_READ, meta


def read_df(f: BinaryIO) -> Tuple[Dict, pd.DataFrame]:
    state = State.RELATION
    meta = {
        'attributes': []
    }

    for line in f:
        state, meta = _process_line(line.decode('utf-8').strip(), state, meta)
        if state == State.DATA_TAG_READ:
            break

    return meta, pd.read_csv(
        f,
        header=None,
        names=[a.name for a in meta['attributes']],
        skipinitialspace=True,
    )


def read_dataset(fn: str) -> Tuple[Dict, pd.DataFrame]:
    with zf.ZipFile(fn, 'r') as z:
        dataset_fn = next(fn for fn in z.namelist() if fn.endswith('.dat'))
        with z.open(dataset_fn, 'r') as d:
            meta, df = read_df(d)
            target = meta['outputs']
            df[target] = df[target].str.strip()
    return meta, df
