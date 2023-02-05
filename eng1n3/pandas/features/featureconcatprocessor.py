"""
Feature Processor for the creation of FeatureConcat features.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List, TypeVar

from f3atur3s import Feature, FeatureConcat, FeatureTypeInteger

from ..dataframebuilder import FeatureProcessor
from ..common.data import pandas_type
from ..common.exception import EnginePandasException


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
