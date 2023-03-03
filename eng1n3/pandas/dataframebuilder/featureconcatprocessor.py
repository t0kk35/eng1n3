"""
Feature Processor for the creation of FeatureConcat features.
(c) 2023 tsm
"""
import pandas as pd
from typing import List

from f3atur3s import Feature, FeatureConcat

from .dataframebuilder import FeatureProcessor


class FeatureConcatProcessor(FeatureProcessor[FeatureConcat]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureConcatProcessor, self).__init__(FeatureConcat, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        kwargs = {
            f.name: df[f.base_feature.name].astype(str) + df[f.concat_feature.name].astype(str)
            for f in self.features
        }
        # Apply concatenations
        df = df.assign(**kwargs)
        return df
