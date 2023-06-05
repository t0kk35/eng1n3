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
    def __init__(self, cls: Type[T], tensor_definition: TensorDefinition, inference: bool):
        self._tensor_definition = tensor_definition
        features = [f for f in tensor_definition.features]
        self._feature: T = FeatureHelper.filter_feature(cls, features)[0]
        self._inference = inference

    @property
    def tensor_definition(self) -> TensorDefinition:
        return self._tensor_definition

    @property
    def feature(self) -> T:
        return self._feature

    @property
    def inference(self) -> bool:
        return self._inference

    @abstractmethod
    def process(self, df: pd.DataFrame, time_field: Feature, num_threads: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def df_features(self) -> List[Feature]:
        pass
