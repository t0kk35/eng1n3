"""
Definition of the base features types. These are all helper or abstract classes
(c) 2023 tsm
"""
import pandas as pd
from typing import Dict, Type, List

from f3atur3s import Feature, FeatureHelper, FeatureSource, FeatureGrouper

from eng1n3.pandas.common.exception import EnginePandasException

from eng1n3.pandas.features.processor import FeatureProcessor
from eng1n3.pandas.features.featuresourceprocessor import FeatureSourceProcessor
from eng1n3.pandas.features.featuregrouperprocessor import FeatureGrouperProcessor


class DataFrameBuilder:
    def __init__(self, features: List[Feature], file: str, delimiter: chr, quote: chr, num_threads: int,
                 time_feature: Feature, inference: bool):
        self._feature_processors: Dict[Type[Feature], FeatureProcessor] = {}
        self._register_processors(features, file, delimiter, quote, num_threads, time_feature, inference)
        self._val_check_known_func(features, self._feature_processors)

    def _register_processors(self, features: List[Feature], file: str, delimiter: chr, quote: chr, num_threads: int,
                             time_feature: Feature, inference: bool):
        self._feature_processors[FeatureSource] = FeatureSourceProcessor(features, file, delimiter, quote, inference)
        self._feature_processors[FeatureGrouper] = FeatureGrouperProcessor(
            features, num_threads, time_feature, inference
        )

    @staticmethod
    def _val_check_known_func(features: List[Feature], processors: Dict[Type[Feature], FeatureProcessor]) -> None:
        """
        Validation function to see if we know how to build all the features.

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
                f'{[field.name for field in unknown_proc]}'
            )

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        for proc in self._feature_processors.values():
            if len(proc.features) != 0:
                df = proc.process(df)
        return df
