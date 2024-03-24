import logging
import openml

LOGGER = logging.getLogger(__name__)

NUMBER_OF_CLASSES = "2"
NUMBER_OF_INSTANCES = "10000"
NUMBER_OF_FEATURES = "500"

def get_datasets():
    LOGGER.info(f"Fetching datasets using filter: \"NumberOfClasses == {NUMBER_OF_CLASSES} & NumberOfInstances  <= {NUMBER_OF_INSTANCES} & NumberOfFeatures <= {NUMBER_OF_FEATURES}\"")
    datalist_fetched = openml.datasets.list_datasets(output_format="dataframe")
    datasets = datalist_fetched.query(f"NumberOfClasses == {NUMBER_OF_CLASSES} & NumberOfInstances  <= {NUMBER_OF_INSTANCES} & NumberOfFeatures <= {NUMBER_OF_FEATURES}") 
    LOGGER.info(f"Number of datasets: {len(datasets)}")
    return datasets