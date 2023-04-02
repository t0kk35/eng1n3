"""
Feature Processor for the creation of FeatureNormaliseStandard features.
(c) 2023 tsm
"""
import logging
import numpy as np
import pandas as pd
from typing import List

from f3atur3s import Feature, FeatureNormalizeStandard

from ..common.data import pandas_type
from .featurenormalizeprocessor import FeatureNormalizeProcessor

logger = logging.getLogger(__name__)


class FeatureNormalizeStandardProcessor(FeatureNormalizeProcessor[FeatureNormalizeStandard]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureNormalizeStandardProcessor, self).__init__(FeatureNormalizeStandard, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # First Create a dictionary with mappings of normalize scale features. Run all at once at the en
        kwargs = {}
        for f in self.features:
            fn = f.name
            bfn = f.base_feature.name
            log_fn = self.get_log_fn(f)
            t = np.dtype(pandas_type(f))
            if not self.inference:
                f.mean = df[bfn].mean() if log_fn is None else log_fn(df[bfn]+f.delta).mean()
                f.stddev = df[bfn].std() if log_fn is None else log_fn(df[bfn]+f.delta).std()
            logger.info(f'Create {fn} Standard {bfn} feature. Mean {f.mean:.2f} Std {f.stddev:.2f}')
            if log_fn is None:
                kwargs[fn] = ((df[bfn] - f.mean) / f.stddev).astype(t)
            else:
                kwargs[fn] = ((log_fn(df[bfn]+f.delta) - f.mean) / f.stddev).astype(t)

        # Update the Pandas dataframe. All normalizations are applied at once.
        df = df.assign(**kwargs)
        # Return the Pandas dataframe
        return df
