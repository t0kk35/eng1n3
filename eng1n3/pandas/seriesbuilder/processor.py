"""
Definition of the base -abstract- FeatureProcessor class to build series
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from typing import Type, List, TypeVar, Generic

from f3atur3s import Feature, FeatureHelper

T = TypeVar('T', bound=Feature)


class SeriesProcessor(Generic[T], ABC):
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
    def process(self) -> np.ndarray:
        pass
