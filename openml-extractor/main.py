import logging
from dataset_datasource import get_datasets
from learn_datasource import get_f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

if __name__ == '__main__':
    datasets_df = get_datasets()
    for dataset_data in datasets_df.iterrows():
        get_f1_score(dataset_data=dataset_data)
