"""
Feature Processor for the creation of FeatureRation features.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureRatio

from .dataframebuilder import FeatureProcessor
from ..common.data import pandas_type


class FeatureRatioProcessor(FeatureProcessor[FeatureRatio]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureRatioProcessor, self).__init__(FeatureRatio, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add Ratio features. Simple division with some logic to avoid errors and 0 division. Note that pandas return
        # inf if the denominator is 0, and nan if both the numerator and the denominator are 0.
        # Do all ratios in one go using assign
        kwargs = {
            f.name:
                df[f.base_feature.name].
                div(df[f.denominator_feature.name]).
                replace([np.inf, np.nan], 0).
                astype(np.dtype(pandas_type(f)))
            for f in self.features
        }
        # Apply concatenations
        df = df.assign(**kwargs)

        return df
