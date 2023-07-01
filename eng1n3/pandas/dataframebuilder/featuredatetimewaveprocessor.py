"""
Feature Processor for the creation of FeatureDateTimeWave features.
(c) 2023 tsm
"""
import math

import numpy as np
import pandas as pd
from typing import List

from f3atur3s import Feature, FeatureDateTimeWave

from .featureprocessor import FeatureProcessor
from ..common.data import pandas_type


class FeatureDateTimeWaveProcessor(FeatureProcessor[FeatureDateTimeWave]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureDateTimeWaveProcessor, self).__init__(FeatureDateTimeWave, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add the wave features
        # Each frequency will do sin(x / period * 2**(frequency+1) * pi) and cos(x / period * 2**(frequency+1) * pi)
        kwargs = {}
        for f in self.features:
            kwargs.update({
                f'{f.name}{f.delimiter}{fn.__name__}{f.delimiter}{fq}': fn(
                    df[f.base_feature.name].dt.strftime(f.format).astype(int) / f.period * math.pow(2, fq+1) * np.pi
                )
                for fq in range(f.frequencies) for fn in (np.sin, np.cos)
            })
        # Apply formatting
        df = df.assign(**kwargs)
        return df
