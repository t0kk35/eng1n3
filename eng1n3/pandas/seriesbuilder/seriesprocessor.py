"""
Definition of the base -abstract- FeatureProcessor class to build series
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from typing import Type, List, TypeVar, Generic, Callable, Tuple

from f3atur3s import Feature, FeatureHelper, TensorDefinition

from ...common.tensorinstance import TensorInstanceNumpy

T = TypeVar('T', bound=Feature)


class SeriesProcessor(Generic[T], ABC):
    def __init__(self, cls: Type[T], target_tensor_def: Tuple[TensorDefinition, ...], inference: bool):
        self._target_tensor_def = target_tensor_def
        features = [f for td in target_tensor_def for f in td.features]
        self._features: List[T] = FeatureHelper.filter_feature(cls, features)
        self._inference = inference

    @property
    def target_tensor_def(self) -> Tuple[TensorDefinition, ...]:
        return self._target_tensor_def

    @property
    def features(self) -> List[T]:
        return self._features

    @property
    def inference(self) -> bool:
        return self._inference

    @abstractmethod
    def process(self, df: pd.DataFrame, time_field: Feature, num_threads: int) -> TensorInstanceNumpy:
        pass

    @abstractmethod
    def get_df_features(self) -> List[Feature]:
        pass
