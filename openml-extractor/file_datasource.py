import logging
import csv
import os

LOGGER = logging.getLogger(__name__)
BASE_PATH = "./datasets"

CSV_HEADERS = ['did', 'dataset_name', 'f1_score_mean_lr', 'attr_to_inst', 'cat_to_num', 'freq_class.mean', 'freq_class.sd', 'inst_to_attr', 'nr_attr', 'nr_bin', 'nr_cat', 'nr_class', 'nr_inst', 'nr_num', 'num_to_cat']

    
def append_csv(filename, data, headers):
    def create_base_dir():
        if not os.path.exists(BASE_PATH):
            LOGGER.info(f"BASE_PATH does not exists. Creating {BASE_PATH}...")
            os.makedirs(BASE_PATH)
    create_base_dir()
    filename = filename + ".csv"
    path_file = os.path.join(BASE_PATH, filename)
    file_exists = os.path.exists(path_file)
    mode = 'a' if file_exists else 'w'

    with open(path_file, mode, newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(headers)
    
        for linha in data:
            csv_writer.writerow(linha)
    LOGGER.info(f"added {data} into {filename}")

