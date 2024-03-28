import logging
from file_datasource import append_csv
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from metalearn_datasource import get_meta_features


LOGGER = logging.getLogger(__name__)

MAX_ITER_LOGISTIC_REGRESSION = 1000000000


def get_f1_score(dataset_data):
    try:
        csv_data = []
        did = dataset_data[1]['did']
        dataset_name = dataset_data[1]['name']
        dataset = fetch_openml(data_id=did)
        X = pd.get_dummies(dataset.data, drop_first=False)
        y = dataset.target

        logistic_model = LogisticRegression(max_iter=MAX_ITER_LOGISTIC_REGRESSION)
        
        scores = cross_val_score(logistic_model, X, y, cv=5, scoring='f1_weighted')
        f1_score_mean = scores.mean()
        meta_feature_data = get_meta_features(data=X, target=y)
        meta_features_labels = meta_feature_data[0]
        
        csv_headers = ['did', 'dataset_name']
        csv_headers.extend(meta_features_labels)
        csv_headers.append('f1_score_mean')
        
        meta_features = meta_feature_data[1]
        csv_data = [did, dataset_name]
        csv_data.extend(meta_features)
        csv_data.append(f1_score_mean)
        
        append_csv(filename="metadataset", data=[csv_data], headers=csv_headers)
    except:
        pass
        # LOGGER.error(e)