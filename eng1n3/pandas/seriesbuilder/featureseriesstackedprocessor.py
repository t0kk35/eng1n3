"""
Definition of FeatureSeriesStacked SeriesProcessor class.
(c) 2023 tsm
"""
import logging

import pandas as pd
import numpy as np
import numba as nb
from numba import jit
import multiprocessing as mp
from functools import partial
from typing import List, Tuple

from f3atur3s import Feature, FeatureSeriesStacked, FeatureExpressionSeries, TensorDefinition, LearningCategory
from f3atur3s import FeatureType
from f3atur3s import FeatureHelper

from .seriesprocessor import SeriesProcessor
from ..common.data import pandas_type
from ...common.tensorinstance import TensorInstanceNumpy

logger = logging.getLogger(__name__)


class FeatureSeriesStackedProcessor(SeriesProcessor[FeatureSeriesStacked]):

    @property
    def df_features(self) -> List[Feature]:
        f_features = FeatureHelper.filter_not_feature(FeatureExpressionSeries, self.feature.series_features)
        # Add all parameter features
        f_features.extend([
            p for f in FeatureHelper.filter_feature(FeatureExpressionSeries, self.feature.series_features)
            for p in f.param_features
        ])
        f_features.append(self.feature.key_feature)
        return f_features

    def __init__(self, tensor_definition: TensorDefinition, inference: bool):
        super(FeatureSeriesStackedProcessor, self).__init__(FeatureSeriesStacked, tensor_definition, inference)

    def process(self, df: pd.DataFrame, time_feature: Feature, num_threads: int) -> Tuple[np.ndarray, ...]:

        logger.info(f'Start creating stacked series for Target Tensor Definition'
                    f'{self.tensor_definition.name} using {num_threads} process(es)')

        key_feature = self.feature.key_feature

        if num_threads == 1:
            df.sort_values(by=[key_feature.name, time_feature.name], ascending=True, inplace=True)
            # Keep the original index. We'll need it to restore the original order of the input data.
            indexes = df.index
            series = self._process_key_stacked(df, key_feature,  self.feature)
        else:
            # MultiThreaded processing, to be implemented
            # Now stack the data....
            indexes = df.index

            key_function = partial(
                self._process_key_stacked,
                time_feature=time_feature,
                key_feature=key_feature,
                features=self.features
            )
            with mp.Pool(num_threads) as p:
                series = p.map(key_function, [rows for _, rows in df.groupby(key_feature.name)])
                series = [s for keys in series for s in keys]

        # Need to sort to get back in the order of the index
        series = series[indexes.argsort()]
        logger.info(f'Returning series of type {str(series.dtype)}.')
        # Turn it into a NumpyList
        # series = NumpyList(series)
        # Don't forget to set the Rank and shape
        self.tensor_definition.rank = 3
        self.tensor_definition.shapes = [(-1, *series.shape[1:])]
        logger.info(f'Series Shape={self.tensor_definition.shapes}')

        return series

    @staticmethod
    def _process_key_stacked(rows: pd.DataFrame, key_feature: Feature, s_feature: FeatureSeriesStacked) -> np.ndarray:

        series_features = s_feature.series_features
        # Enrich the series. Run the FeatureSeriesExpression logic. Note this is a list of lists which we flatten
        sf: List[FeatureExpressionSeries] = FeatureHelper.filter_feature(FeatureExpressionSeries, series_features)

        for f in sf:
            t = pandas_type(f)
            rows[f.name] = f.expression(rows[[p.name for p in f.param_features]]).astype(t)

        # Convert everything to numpy for performance. This creates a numpy per each LC, with all feature of that LC.
        np_series = rows[[f.name for f in series_features]].to_numpy(pandas_type(s_feature))

        same_key = pd.concat((
            pd.Series([True]),
            rows[key_feature.name].iloc[1:].reset_index(drop=True).
            eq(rows[key_feature.name].iloc[:-1].reset_index(drop=True))
        )).to_numpy()

        np_out = _numba_process_stacked_keys(same_key, np_series, s_feature.series_depth)
        return np_out


@jit(nopython=True, cache=True)
def _numba_process_stacked_keys(same_key: np.ndarray, base_values: np.ndarray, window: int) -> np.ndarray:
    """
    Numba jit-ed function to build stacked series. We're using numba here because we iterate over the entire input.
    There might be a way to vectorize this. But I have not found it.

    Args:
        same_key : A Numpy array of type bool and shape (#row_to_process). It contains True if the row at a specific
            location contained the same key as the previous row. It is used to reset the profile element to
            zeros.
        base_values: These are the values that need to be stacked. They are the original data-frame in an array format
        window: Int value containing the window for  the lists (i.e. series depths we want to use).

    Returns:
        A numpy array containing the series
    """
    # Allocate output structure
    np_out = np.zeros((base_values.shape[0], window, base_values.shape[1]), dtype=base_values.dtype)

    # Keep a memory for the count of the first records we processed for a given key. We use it to avoid reading into
    # the previous key
    i_key_mem = 0

    for i in range(base_values.shape[0]):
        if not same_key[i]:
            i_key_mem = i
        s = base_values[max(i_key_mem, i - window + 1):i + 1]
        # Pre-Pad if incomplete. I.e. There were less than length rows before this row.
        np_out[i, :, :] = np.concatenate(
            (np.zeros((window - s.shape[0], s.shape[1]), dtype=s.dtype), s)
        ) if s.shape[0] < window else s

    return np_out
