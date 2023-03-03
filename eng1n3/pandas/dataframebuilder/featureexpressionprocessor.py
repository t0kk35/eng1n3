"""
Feature Processor for the creation of FeatureExpression features.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureExpression

from .dataframebuilder import FeatureProcessor
from ..common.data import pandas_type


class FeatureExpressionProcessor(FeatureProcessor[FeatureExpression]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureExpressionProcessor, self).__init__(FeatureExpression, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add the expression fields. Just call the expression with the parameter names. Use vectorization. Second best
        # to Native vectorization and faster than apply.
        kwargs = {
            f.name:
                np.vectorize(f.expression, otypes=[pandas_type(f, read=False)])(df[[f.name for f in f.param_features]])
            for f in self.features
        }
        # Apply concatenations
        df = df.assign(**kwargs)
        return df
