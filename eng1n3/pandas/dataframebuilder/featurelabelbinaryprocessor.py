"""
Feature Processor for the creation of FeatureLabelBinary features.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureLabelBinary

from .dataframebuilder import FeatureProcessor
from ..common.data import pandas_type
from ..common.exception import EnginePandasException


class FeatureLabelBinaryProcessor(FeatureProcessor[FeatureLabelBinary]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureLabelBinaryProcessor, self).__init__(FeatureLabelBinary, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        for f in self.features:
            t = np.dtype(pandas_type(f))
            df[f.name] = df[f.base_feature.name].copy().astype(t)
        return df

    @staticmethod
    def _val_int_is_binary(df: pd.DataFrame, feature: FeatureLabelBinary):
        """
        Validation routine that checks if a value only contains 0 and 1. A binary label should only have values
        0 and 1.

        Args:
            df: The Pandas DataFrame to check that should contain the label.
            feature: An instance of FeatureLabelBinary.

        Returns:
            None

        Raises:
            EnginePandasException: If there are values other than 0 and 1 as feature value in the DataFrame.
        """
        u = sorted(list(pd.unique(df[feature.base_feature.name])))
        if not u == [0, 1]:
            raise EnginePandasException(
                f'Binary Feature <{feature.name}> should only contain values 0 and 1 . Found values {u}')
