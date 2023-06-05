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
from ..common.exception import EnginePandasException
from ...common.tensorinstance import TensorInstanceNumpy


class SeriesBuilder:
    def __init__(self, target_tensor_def: Tuple[TensorDefinition, ...], inference: bool, num_threads: int):
        self._target_tensor_def = target_tensor_def
        self._inference = inference
        self._num_threads = num_threads
        self._feature_processors = SeriesBuilder._processors(target_tensor_def, inference)

    @staticmethod
    def _processors(target_tensor_def: Tuple[TensorDefinition, ...], inference: bool) -> Dict[str, SeriesProcessor]:
        procs = {}
        for td in target_tensor_def:
            feature = td.features[0]
            if isinstance(feature, FeatureSeriesStacked):
                procs[td.name] = FeatureSeriesStackedProcessor(td, inference)
            else:
                raise EnginePandasException(
                    f'Could not find Series processor for {type(feature)}'
                )
        return procs

    @property
    def target_tensor_def(self) -> Tuple[TensorDefinition, ...]:
        return self._target_tensor_def

    @property
    def num_threads(self) -> int:
        return self._num_threads

    @property
    def df_features(self) -> List[Feature]:
        """
        Return all the features the series builder will need to create the series. These features will be fed in the
        dataframe provided to the build call.

        Returns:
            A list of features that will be built with the df_from_csv call and provided to the build method.
        """
        return list(set([f for proc in self._feature_processors.values() for f in proc.df_features]))

    @property
    def processors(self) -> Dict[str, SeriesProcessor]:
        return self._feature_processors

    def build(self, td: TensorDefinition, df: pd.DataFrame, time_feature: Feature) -> np.ndarray:
        """
        Validation function to see if we know how to build all the series.

        Args:
            td: The TensorDefinition that needs to be built. This tensor definition has to be one that was in the
                target_tensor_def parameter of the Series construction call.
            df: A Pandas Dataframe that contains the base features needed to create the Series.
            time_feature: A Feature object that will be used as time feature for the processing of the Series.

        Returns:
             A Numpy Array containing the Series features.
        """
        s = self.processors.get(td.name).process(df, time_feature, self.num_threads)
        return s

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
