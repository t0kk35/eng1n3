"""
Definition of the base features types. These are all helper or abstract classes
(c) 2023 tsm
"""
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from numba import jit
from functools import partial
from itertools import groupby
from collections import OrderedDict
from typing import Tuple, List, Dict

from f3atur3s import Feature, FeatureGrouper

from .dataframebuilder import FeatureProcessor
from ...profile.profilenumpy import profile_time_logic, profile_aggregate, profile_contrib, ProfileNumpy
from ..common.data import pandas_type

logger = logging.getLogger(__name__)


class FeatureGrouperProcessor(FeatureProcessor[FeatureGrouper]):
    def __init__(self, features: List[Feature], num_threads: int, time_feature: Feature, inference: bool):
        super(FeatureGrouperProcessor, self).__init__(FeatureGrouper, features, inference)
        self._num_threads = num_threads
        self._time_feature = time_feature

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.features) == 0:
            return df
        # Group per Group feature. i.e. per key.
        group_dict: Dict[Feature, List[FeatureGrouper]] = OrderedDict(
            sorted([
                (g, list(gf)) for g, gf in groupby(
                    sorted(self.features, key=lambda x: x.group_feature), lambda x: x.group_feature
                )
            ], key=lambda x: x[0]))

        for g, gf in group_dict.items():
            logger.info(f'Start creating aggregate grouper feature for <{g.name}> ' +
                        f'using {self._num_threads} process(es)')

            # Single process processing
            if self._num_threads == 1:
                df.sort_values(by=[g.name, self._time_feature.name], ascending=True, inplace=True)
                # Set-up Numpy profile and variables. First row is always True.
                p = ProfileNumpy(gf)
                same_key = pd.concat((
                    pd.Series([True]),
                    df[g.name].iloc[1:].reset_index(drop=True).eq(df[g.name].iloc[:-1].reset_index(drop=True))
                ))
                # Run numba jit-ed loop over the row in the df and process.
                out = _numba_process_grouper(
                    same_key.to_numpy(),
                    df[[f.name for f in p.base_features]].to_numpy(),
                    df[[f.name for f in p.filter_features]].to_numpy(),
                    ProfileNumpy.get_deltas(df, self._time_feature), p.base_filters, p.feature_filters,
                    p.timeperiod_filters, p.aggregator_indexes, p.filter_indexes, p.time_windows, len(gf), p.array_shape
                )
                ags = pd.DataFrame(
                    {f.name: pd.Series(np.squeeze(out[:, i]), index=df.index).astype(pandas_type(f))
                     for i, f in enumerate(gf)}
                )
                df = pd.concat([df, ags], axis=1)
            # Multi process processing
            else:
                key_function = partial(
                    _process_grouper_key,
                    group_features=gf,
                    time_feature=self._time_feature
                )
                with mp.Pool(self._num_threads) as p:
                    dfs = p.map(key_function, [rows for _, rows in df.groupby(g.name)])
                df = pd.concat(dfs, axis=0)

            logger.info(f'Start creating aggregate grouper features for <{g.name}> ')

        # Restore Original Sort
        df.sort_index(inplace=True)
        return df


def _process_grouper_key(rows: pd.DataFrame,
                         group_features: List[FeatureGrouper],
                         time_feature: Feature) -> pd.DataFrame:
    rows.sort_values(by=time_feature.name, ascending=True, inplace=True)
    p = ProfileNumpy(group_features)
    # Run numba jit-ed loop over the row in the df and process.
    out = _numba_process_grouper(
        np.ones(len(rows), dtype=np.bool),
        rows[[f.name for f in p.base_features]].to_numpy(),
        rows[[f.name for f in p.filter_features]].to_numpy(),
        ProfileNumpy.get_deltas(rows, time_feature), p.base_filters, p.feature_filters, p.timeperiod_filters,
        p.aggregator_indexes, p.filter_indexes, p.time_windows, len(group_features), p.array_shape
    )
    ags = pd.DataFrame(
        {f.name: pd.Series(np.squeeze(out[:, i]), index=rows.index).astype(pandas_type(f))
         for i, f in enumerate(group_features)}
    )
    return pd.concat([rows, ags], axis=1)


@jit(nopython=True, cache=True)
def _numba_process_grouper(same_key: np.ndarray, base_values: np.ndarray, filter_values: np.ndarray,
                           deltas: np.ndarray, b_flt: np.ndarray, f_flt: np.ndarray, tp_flt: np.ndarray,
                           a_ind: np.ndarray, f_ind: np.ndarray, tw: np.ndarray, group_feature_cnt: int,
                           pe_shape: Tuple[int, int, int]) -> np.array:
    """
    A Numba jit-ed function to that creates grouper features for each row in a Numpy array.

    Args:
        same_key (np.ndarray): A Numpy array of type bool and shape (#row_to_process). It contains True if the row at
            a specific location contained the same key as the previous row. It is used to reset the profile element to
            zeros.
        base_values (np.ndarray): A Numpy array of type float. It contains the values of the base feature that need
            to contribute to the profile. It has shape (#rows_to_process X #base_features)
        filter_values (np.ndarray): A Numpy array of type int. Has the values for the filter features that are used
            in the profile. It has shape (#rows_to_process X #filter_features)
        deltas (np.ndarray): A Numpy array of type int with the deltas for each of the TimePeriods. For each row to
            process it contains the difference in time compare to the previous row.
            It has shape (#rows_to_process X #time_periods). It can be fetched with the `ProfileNumpy.get_deltas` method
        b_flt (np.ndarray): A numpy array filter of type np.bool. Is an array that contains the filters for each
            base_feature in the profile. There is a row for each base filter, the columns on one specific row contain a
            filter that filters out the profile elements that should be updated by the respective base_feature.
            It has shape (#base_feature X #elements_in_profile)
            Values should be fetched with the `ProfileNumpy.base_filters` property
        f_flt (np.ndarray): A numpy array of type bool with shape (#group_features X #profile_elements). Each row
            contains a filter that can be used to select the correct profile element for that feature from the profile
            element array. Values should be fetched with the `ProfileNumpy.filter_features` property
        tp_flt (np.ndarray): An ndarray of type int. It has shape (#time_periods X #profile_element). It contains
            a row for each TimePeriod object. The row is a filter that filters out the elements using the respective
            TimePeriod object. Can be created with the `ProfileNumpy.timeperiod_filters` property
        a_ind (np.ndarray): A numpy array of type int with shape (#group_features). Each row contains the `key`/id
            of an Aggregator object for a specific group feature of this profile. Values should be fetched with the
            `ProfileNumpy.aggregator_indexes` property
        f_ind (np.ndarray) : A numpy array that holds an index to the filter that needs to be applied to each
            element in the profile. Values should be fetched with the `ProfileNumpy.filter_indexes` property
        tw (np.ndarray): A numpy array of type int with shape (#group_features). Each row the time window
            to be applied to a specific group feature of this profile. Values should be fetched with the
            `ProfileNumpy.time_windows` property
        group_feature_cnt (int): Number of FeatureGroupers used in the creation of the profile
        pe_shape Tuple(int, int, int): The shape of the profile element Numpy array. It has 3 dimensions. Values can
            be fetched with the `ProfileNumpy.array_shape` property.
    Returns:
        An Numpy array containing all the aggregate values for GrouperFeatures in this profile.
            It has shape (#rows_to_process X #grouper_features)
    """
    out = np.zeros((base_values.shape[0], group_feature_cnt))
    p = np.zeros(pe_shape)
    for i in range(base_values.shape[0]):
        if i > 0 and not same_key[i]:
            # Reset profile values
            p.fill(0.0)
        else:
            profile_time_logic(tp_flt, deltas[i], p)
        profile_contrib(b_flt, f_ind, base_values[i], filter_values[i], p)
        out[i, :] = profile_aggregate(f_flt, a_ind, tw, p)
    return out
