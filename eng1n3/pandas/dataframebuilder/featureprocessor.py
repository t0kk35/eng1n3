"""
Definition of the base -abstract- FeatureProcessor class
(c) 2023 tsm
"""
import pandas as pd
from abc import ABC, abstractmethod
from typing import Type, List, TypeVar, Generic

from f3atur3s import Feature, FeatureHelper

T = TypeVar('T', bound=Feature)


class FeatureProcessor(Generic[T], ABC):
    def __init__(self, cls: Type[T], features: List[Feature], inference: bool):
        self._features: List[T] = FeatureHelper.filter_feature(cls, features)
        self._inference = inference

    @property
    def features(self) -> List[T]:
        return self._features

    @property
    def inference(self) -> bool:
        return self._inference

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
