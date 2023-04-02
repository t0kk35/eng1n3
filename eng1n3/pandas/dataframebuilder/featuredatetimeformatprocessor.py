"""
Feature Processor for the creation of FeatureDateTimeFormat features.
(c) 2023 tsm
"""
import pandas as pd
from typing import List

from f3atur3s import Feature, FeatureDateTimeFormat

from .featureprocessor import FeatureProcessor
from ..common.data import pandas_type


class FeatureDateTimeFormatProcessor(FeatureProcessor[FeatureDateTimeFormat]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureDateTimeFormatProcessor, self).__init__(FeatureDateTimeFormat, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add the binning features
        kwargs = {
            f.name: df[f.base_feature.name].dt.strftime(f.format).astype(pandas_type(f))
            for f in self.features
        }
        # Apply formatting
        df = df.assign(**kwargs)
        return df
