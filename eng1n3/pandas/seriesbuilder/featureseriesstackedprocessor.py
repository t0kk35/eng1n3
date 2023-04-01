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

    def get_df_features(self) -> List[Feature]:
        f_features = FeatureHelper.filter_not_feature(FeatureExpressionSeries, self.features[0].series_features)
        # Add all parameter features
        f_features.extend([
            p for f in FeatureHelper.filter_feature(FeatureExpressionSeries, self.features[0].series_features)
            for p in f.param_features
        ])
        f_features.append(self.features[0].key_feature)
        return f_features

    def __init__(self, target_tensor_def: Tuple[TensorDefinition, ...], inference: bool):
        super(FeatureSeriesStackedProcessor, self).__init__(FeatureSeriesStacked, target_tensor_def, inference)

    def process(self, df: pd.DataFrame, time_feature: Feature, num_threads: int) -> Tuple[np.ndarray, ...]:

        logger.info(f'Start creating stacked series for Target Tensor Definitions '
                    f'{[td.name for td in self.target_tensor_def]} using {num_threads} process(es)')

        key_feature = self.features[0].key_feature

        if num_threads == 1:
            df.sort_values(by=[key_feature.name, time_feature.name], ascending=True, inplace=True)
            # Keep the original index. We'll need it to restore the original order of the input data.
            indexes = df.index
            series = self._process_key_stacked(df, key_feature,  self.features)
        else:
            # MultiThreaded processing, to be implemented
            # Now stack the data....
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
        series = [s[indexes.argsort()] for s in series]

        logger.info(f'Returning series of types {[str(s.dtype) for s in series]}.')
        # Turn it into a NumpyList
        # series = NumpyList(series)
        # Don't forget to set the Rank and shape
        for td, s in zip(self.target_tensor_def, series):
            td.rank = 3
            td.shapes = (-1, *s.shape[1:])
        logger.info(f'Series Shapes={[td.shapes for td in self.target_tensor_def]}')

        return tuple(series)

    @staticmethod
    def _process_key_stacked(rows: pd.DataFrame, key_feature: Feature,
                             features: List[FeatureSeriesStacked]) -> List[np.ndarray]:

        series_features = [f.series_features for f in features]
        # Enrich the series. Run the FeatureSeriesExpression logic. Note this is a list of lists which we flatten
        sf: List[FeatureExpressionSeries] = [
            f for fs in [
                FeatureHelper.filter_feature(FeatureExpressionSeries, f_lst) for f_lst in series_features
            ] for f in fs
        ]
        for f in sf:
            t = pandas_type(f)
            rows[f.name] = f.expression(rows[[p.name for p in f.param_features]]).astype(t)

        # Convert everything to numpy for performance. This creates a numpy per each LC, with all feature of that LC.
        np_series = tuple([rows[[f.name for f in f.series_features]].to_numpy(pandas_type(f)) for f in features])

        same_key = pd.concat((
            pd.Series([True]),
            rows[key_feature.name].iloc[1:].reset_index(drop=True).
            eq(rows[key_feature.name].iloc[:-1].reset_index(drop=True))
        )).to_numpy()

        np_out = _numba_process_stacked_keys(same_key, np_series, tuple([f.series_depth for f in features]))
        return np_out


@jit(nopython=True, cache=True)
def _numba_process_stacked_keys(same_key: np.ndarray, base_values: Tuple[np.ndarray, ...],
                                windows: Tuple[int, ...]) -> List[np.ndarray]:
    """
    Numba jit-ed function to build stacked series. We're using numba here because we iterate over the entire input.
    There might be a way to vectorize this. But I have not found it.

    Args:
        same_key : A Numpy array of type bool and shape (#row_to_process). It contains True if the row at a specific
            location contained the same key as the previous row. It is used to reset the profile element to
            zeros.
        base_values: These are the values that need to be stacked. They are the original data-frame in an array format
        windows: A tuple of numpy array containing the windows for each of the lists (i.e. series depths we
            want to use).

    Returns:
        A list of numpy arrays.
    """
    # Allocate output structure
    np_out = [np.zeros((s.shape[0], windows[i], s.shape[1])) for i, s in enumerate(base_values)]

    # Keep a memory for the count of the first records we processed for a given key. We use it to avoid reading into
    # the previous key
    i_key_mem = 0

    for i in range(base_values[0].shape[0]):
        if not same_key[i]:
            i_key_mem = i
        for j in range(len(np_out)):
            s = base_values[j][max(i_key_mem, i - windows[j] + 1):i + 1]
            # Pre-Pad if incomplete. I.e. There were less than length rows before this row.
            np_out[j][i, :, :] = np.concatenate(
                (np.zeros((windows[j] - s.shape[0], s.shape[1]), dtype=s.dtype), s)
            ) if s.shape[0] < windows[j] else s

    return np_out
