"""
Definition of FeatureSeriesStacked SeriesProcessor class.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureSeriesStacked

from .processor import SeriesProcessor
from ..common.data import pandas_type


class FeatureSeriesStackedProcessor(SeriesProcessor[FeatureSeriesStacked]):

    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureSeriesStackedProcessor, self).__init__(FeatureSeriesStacked, features, inference)

    def process(self) -> np.ndarray:
        pass
