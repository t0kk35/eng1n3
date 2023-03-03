"""
Definition of the base features types. These are all helper or abstract classes
(c) 2023 tsm
"""
import pandas as pd
from typing import Dict, Type, List

from f3atur3s import Feature, FeatureHelper, FeatureSource, FeatureOneHot, FeatureIndex, FeatureBin, FeatureGrouper
from f3atur3s import FeatureConcat, FeatureExpression, FeatureRatio, FeatureNormalizeScale, FeatureNormalizeStandard
from f3atur3s import FeatureFilter

from ..common.exception import EnginePandasException

from .processor import FeatureProcessor
from .featuresourceprocessor import FeatureSourceProcessor
from .featuregrouperprocessor import FeatureGrouperProcessor
from .featureonehotprocessor import FeatureOneHotProcessor
from .featureindexprocessor import FeatureIndexProcessor
from .featurebinprocessor import FeatureBinProcessor
from .featureconcatprocessor import FeatureConcatProcessor
from .featureexpressionprocessor import FeatureExpressionProcessor
from .featureratioprocessor import FeatureRatioProcessor
from .featurenormalizescaleprocessor import FeatureNormalizeScaleProcessor
from .featurenormalizestandardprocessor import FeatureNormalizeStandardProcessor


class DataFrameBuilder:
    def __init__(self, features: List[Feature], file: str, delimiter: chr, quote: chr, num_threads: int,
                 one_hot_prefix: str, time_feature: Feature, inference: bool):
        self._features = features
        self._file = file
        self._delimiter = delimiter
        self._quote = quote
        self._num_threads = num_threads
        self._one_hot_prefix = one_hot_prefix
        self._time_feature = time_feature
        self._inference = inference
        self._feature_processors = self._register_processors()
        self._val_check_known_func(features, self._feature_processors)

    def _register_processors(self) -> Dict[Type[Feature], FeatureProcessor]:
        return {
            FeatureSource:
                FeatureSourceProcessor(self._features, self._file, self._delimiter, self._quote, self._inference),
            FeatureOneHot:
                FeatureOneHotProcessor(self._features, self._one_hot_prefix, self._inference),
            FeatureIndex:
                FeatureIndexProcessor(self._features, self._inference),
            FeatureBin:
                FeatureBinProcessor(self._features, self._inference),
            FeatureConcat:
                FeatureConcatProcessor(self._features, self._inference),
            FeatureExpression:
                FeatureExpressionProcessor(self._features, self._inference),
            FeatureRatio:
                FeatureRatioProcessor(self._features, self._inference),
            FeatureNormalizeScale:
                FeatureNormalizeScaleProcessor(self._features, self._inference),
            FeatureNormalizeStandard:
                FeatureNormalizeStandardProcessor(self._features, self._inference),
            FeatureGrouper:
                FeatureGrouperProcessor(self._features, self._num_threads, self._time_feature, self._inference)
        }

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
