"""
Definition of the base features types. These are all helper or abstract classes
(c) 2023 tsm
"""
import pandas as pd
import numpy as np

from typing import List, Dict, Type, Callable, Tuple

from f3atur3s import Feature, FeatureSeriesBased, FeatureSeriesStacked, FeatureHelper, FeatureType, TensorDefinition

from .seriesprocessor import SeriesProcessor
from .featureseriesstackedprocessor import FeatureSeriesStackedProcessor
from ...common.tensorinstance import TensorInstanceNumpy


class SeriesBuilder:
    def __init__(self, target_tensor_def: Tuple[TensorDefinition, ...], inference: bool, num_threads: int):
        self._target_tensor_def = target_tensor_def
        self._inference = inference
        self._num_threads = num_threads
        self._feature_processor = SeriesBuilder._processor(target_tensor_def, inference)

    @staticmethod
    def _processor(target_tensor_def: Tuple[TensorDefinition, ...], inference: bool) -> SeriesProcessor:
        feature = target_tensor_def[0].features[0]
        if isinstance(feature, FeatureSeriesStacked):
            return FeatureSeriesStackedProcessor(target_tensor_def, inference)
        else:
            raise EnginePandasException(
                f'Could not find Series processor for {type(Feature)}'
            )

    @property
    def target_tensor_def(self) -> Tuple[TensorDefinition, ...]:
        return self._target_tensor_def

    @property
    def num_threads(self) -> int:
        return self._num_threads

    @property
    def df_features(self) -> List[Feature]:
        return self._feature_processor.get_df_features()

    def get_processor(self, feature_type: Type[FeatureSeriesBased]) -> SeriesProcessor:
        return self._feature_processors[feature_type]

    def build(self, df: pd.DataFrame, time_feature: Feature) -> TensorInstanceNumpy:
        return self._feature_processor.process(df, time_feature, self.num_threads)

    @staticmethod
    def _val_check_known_func(target_tensor_def: Tuple[TensorDefinition, ...],
                              processors: Dict[Type[Feature], SeriesProcessor]) -> None:
        """
        Validation function to see if we know how to build all the series.

        Args:
            target_tensor_def: A tuple of TensorDefinitions to be built.
            processors: Dictionary with all known classes and their respective FeatureProcessors

        Returns:
             None
        """
        features = [f for td in target_tensor_def for f in td.features]
        known_proc = [f for s in processors.keys() for f in FeatureHelper.filter_feature(s, features)]
        unknown_proc = [f for td in target_tensor_def for f in td.features if f not in known_proc]
        if len(unknown_proc) != 0:
            raise EnginePandasException(
                f'Do not know how to build field type. Can not build features: '
                f'{[field.name for field in unknown_proc]} of types: '
                f'{[type(field) for field in unknown_proc]}'
            )
