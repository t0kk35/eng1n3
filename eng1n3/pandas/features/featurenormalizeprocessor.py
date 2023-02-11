"""
Feature Processor base class for FeatureNormalize features. This is an abstract class
(c) 2023 tsm
"""
import numpy as np
from abc import ABC
from typing import Type, List, Optional, Callable

from f3atur3s import Feature, FeatureNormalizeLogBase

from .processor import T
from ..common.exception import EnginePandasException
from ..dataframebuilder import FeatureProcessor


class FeatureNormalizeProcessor(FeatureProcessor[T], ABC):
    def __init__(self, cls: Type[T], features: List[Feature], inference: bool):
        super(FeatureNormalizeProcessor, self).__init__(cls, features, inference)

    @staticmethod
    def get_log_fn(f: FeatureNormalizeLogBase) -> Optional[Callable]:
        if f.log_base is None:
            return None
        if f.log_base == 'e':
            return np.log
        elif f.log_base == '10':
            return np.log10
        elif f.log_base == '2':
            return np.log2
        else:
            raise EnginePandasException(
                f'Problem processing Normalizer feature {f.name}. ' +
                f'Did not find function to calculated log-base {f.log_base}'
            )
