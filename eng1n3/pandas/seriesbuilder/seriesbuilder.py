"""
Definition of the base features types. These are all helper or abstract classes
(c) 2023 tsm
"""
import pandas as pd
import numpy as np

from typing import List, Dict, Type

from f3atur3s import Feature, FeatureSeriesBased, FeatureSeriesStacked

from .processor import SeriesProcessor
from .featureseriesstackedprocessor import FeatureSeriesStackedProcessor


class SeriesBuilder:
    def __init__(self, features: List[FeatureSeriesBased],  inference: bool):
        self._features = features
        self._inference = inference
        self._feature_processors = self._register_processors()
        self._val_check_known_func(features, self._feature_processors)

    def _register_processors(self) -> Dict[Type[Feature], SeriesProcessor]:
        return {
            FeatureSeriesStacked: FeatureSeriesStackedProcessor(self._features, self._inference)
        }

    def build(self, na: np.ndarray) -> np.ndarray:
        for proc in self._feature_processors.values():
            if len(proc.features) != 0:
                na = proc.process()
        return na

    @staticmethod
    def _val_check_known_func(features: List[Feature], processors: Dict[Type[Feature], SeriesProcessor]) -> None:
        """
        Validation function to see if we know how to build all the series.

        Args:
            features: All feature that need to be built.
            processors: Dictionary with all known classes and their respective FeatureProcessors

        Returns:
             None
        """
        known_proc = [f for s in processors.keys() for f in FeatureHelper.filter_feature(s, features)]
        unknown_proc = [f for f in features if f not in known_proc]
        if len(unknown_proc) != 0:
            raise EnginePandasException(
                f'Do not know how to build field type. Can not build features: '
                f'{[field.name for field in unknown_proc]} of types: '
                f'{[type(field) for field in unknown_proc]}'
            )
