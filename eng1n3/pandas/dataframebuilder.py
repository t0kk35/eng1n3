"""
Definition of the base features types. These are all helper or abstract classes
(c) 2023 tsm
"""
import pandas as pd
from typing import Dict, Type, List

from f3atur3s import Feature, FeatureHelper, FeatureSource, FeatureOneHot, FeatureIndex, FeatureBin, FeatureGrouper
from f3atur3s import FeatureConcat, FeatureExpression, FeatureRatio, FeatureNormalizeScale, FeatureNormalizeStandard

from eng1n3.pandas.common.exception import EnginePandasException

from .features.processor import FeatureProcessor
from .features.featuresourceprocessor import FeatureSourceProcessor
from .features.featuregrouperprocessor import FeatureGrouperProcessor
from .features.featureonehotprocessor import FeatureOneHotProcessor
from .features.featureindexprocessor import FeatureIndexProcessor
from .features.featurebinprocessor import FeatureBinProcessor
from .features.featureconcatprocessor import FeatureConcatProcessor
from .features.featureexpressionprocessor import FeatureExpressionProcessor
from .features.featureratioprocessor import FeatureRatioProcessor
from .features.featurenormalizescaleprocessor import FeatureNormalizeScaleProcessor
from .features.featurenormalizestandardprocessor import FeatureNormalizeStandardProcessor


class DataFrameBuilder:
    def __init__(self, features: List[Feature], file: str, delimiter: chr, quote: chr, num_threads: int,
                 one_hot_prefix: str, time_feature: Feature, inference: bool):
        self._feature_processors: Dict[Type[Feature], FeatureProcessor] = {}
        self._register_processors(
            features, file, delimiter, quote, num_threads, one_hot_prefix, time_feature, inference
        )
        self._val_check_known_func(features, self._feature_processors)

    def _register_processors(self, features: List[Feature], file: str, delimiter: chr, quote: chr, num_threads: int,
                             one_hot_prefix: str, time_feature: Feature, inference: bool):
        self._feature_processors[FeatureSource] = FeatureSourceProcessor(features, file, delimiter, quote, inference)
        self._feature_processors[FeatureOneHot] = FeatureOneHotProcessor(features, one_hot_prefix, inference)
        self._feature_processors[FeatureIndex] = FeatureIndexProcessor(features, inference)
        self._feature_processors[FeatureBin] = FeatureBinProcessor(features, inference)
        self._feature_processors[FeatureConcat] = FeatureConcatProcessor(features, inference)
        self._feature_processors[FeatureExpression] = FeatureExpressionProcessor(features, inference)
        self._feature_processors[FeatureRatio] = FeatureRatioProcessor(features, inference)
        self._feature_processors[FeatureNormalizeScale] = FeatureNormalizeScaleProcessor(features, inference)
        self._feature_processors[FeatureNormalizeStandard] = FeatureNormalizeStandardProcessor(features, inference)
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
                f'{[field.name for field in unknown_proc]} of types: '
                f'{[type(field) for field in unknown_proc]}'
            )

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        for proc in self._feature_processors.values():
            if len(proc.features) != 0:
                df = proc.process(df)
        return df
