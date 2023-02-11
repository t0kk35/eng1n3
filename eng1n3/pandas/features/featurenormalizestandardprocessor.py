"""
Feature Processor for the creation of FeatureNormaliseStandard features.
(c) 2023 tsm
"""
import logging
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureNormalizeStandard, FeatureTypeInteger

from ..dataframebuilder import FeatureProcessor
from .featurenormalizeprocessor import FeatureNormalizeProcessor
from ..common.data import pandas_type
from ..common.exception import EnginePandasException

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
            if not self.inference:
                f.mean = df[bfn].mean() if log_fn is None else log_fn(df[bfn]+f.delta).mean()
                f.stddev = df[bfn].std() if log_fn is None else log_fn(df[bfn]+f.delta).std()
            logger.info(f'Create {fn} Standard {bfn} feature. Mean {f.mean:.2f} Std {f.stddev:.2f}')
            if log_fn is None:
                kwargs[fn] = (df[bfn] - f.mean) / f.stddev
            else:
                kwargs[fn] = (log_fn(df[bfn]+f.delta) - f.mean) / f.stddev

        # Update the Pandas dataframe. All normalizations are applied at once.
        df = df.assign(**kwargs)
        # Return the Pandas dataframe
        return df
