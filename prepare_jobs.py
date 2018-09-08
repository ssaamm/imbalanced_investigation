import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os

import common

logger = common.get_logger(__name__)

if __name__ == '__main__':
    logger.info('Reading jobs sheet')
    df = pd.read_csv(os.path.expanduser('~/Downloads/jobs - Sheet1.csv'))
    rated_jobs = df[~pd.isnull(df['Sounds cool'])]

    for klass, name in ((CountVectorizer, 'jobs_wordcount.csv'),
                        (TfidfVectorizer, 'jobs_tfidf.csv')):
        v = klass(max_features=300)
        features = v.fit_transform(rated_jobs['Title']).toarray()

        vect_jobs = pd.DataFrame(features, columns=v.get_feature_names(), index=rated_jobs.index)
        vect_jobs['sounds_cool'] = rated_jobs['Sounds cool']

        logger.info(f'Writing {name}')
        common.write_dataset(vect_jobs, name, target='sounds_cool')
    common.write_tracker()
