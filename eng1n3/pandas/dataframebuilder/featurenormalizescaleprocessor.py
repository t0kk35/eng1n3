"""
Feature Processor for the creation of FeatureNormaliseScale features.
(c) 2023 tsm
"""
import logging
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureNormalizeScale

from ..common.data import pandas_type
from .featurenormalizeprocessor import FeatureNormalizeProcessor

logger = logging.getLogger(__name__)


class FeatureNormalizeScaleProcessor(FeatureNormalizeProcessor[FeatureNormalizeScale]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureNormalizeScaleProcessor, self).__init__(FeatureNormalizeScale, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # First Create a dictionary with mappings of normalize scale features. Run all at once at the end.
        kwargs = {}
        for f in self.features:
            fn = f.name
            bfn = f.base_feature.name
            log_fn = self.get_log_fn(f)
            t = np.dtype(pandas_type(f))
            if not self.inference:
                f.minimum = df[bfn].min() if log_fn is None else log_fn(df[bfn].min()+f.delta)
                f.maximum = df[bfn].max() if log_fn is None else log_fn(df[bfn].max()+f.delta)
            logger.info(f'Create {fn} Scale {bfn}. Min. {f.minimum:.2f} Max. {f.maximum:.2f}')
            if log_fn is None:
                kwargs[fn] = ((df[bfn] - f.minimum) / (f.maximum - f.minimum)).astype(t)
            else:
                kwargs[fn] = ((log_fn(df[bfn]+f.delta) - f.minimum) / (f.maximum - f.minimum)).astype(t)

        # Update the Pandas dataframe. All normalizations are applied at once.
        df = df.assign(**kwargs)
        # Return the Pandas dataframe
        return df
