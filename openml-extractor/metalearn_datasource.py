from pymfe.mfe import MFE
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def get_meta_features(data, target):
    mfe = MFE(groups="all")
    logging.info("starting MFE")
    mfe.fit(data.values, np.array(target.values))
    return mfe.extract()
    