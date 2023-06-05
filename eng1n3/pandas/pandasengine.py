"""
Definition of the Pandas engine. This engine will mainly read files into Pandas and then potentially convert
them to numpy array for modelling/processing.
(c) 2023 tsm
"""
import logging
import pathlib
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import Optional, List, Union, Tuple

from f3atur3s import Feature, FeatureExpander, TensorDefinition, FeatureSeriesBased, FeatureSeriesStacked
from ..common.engine import EngineContext
from ..common.tensorinstance import TensorInstanceNumpy
from .common.exception import EnginePandasException
from .helpers.validation import EnginePandasValidation
from .dataframebuilder.dataframebuilder import DataFrameBuilder
from .seriesbuilder.seriesbuilder import SeriesBuilder

logger = logging.getLogger(__name__)


class EnginePandas(EngineContext):
    """
    Panda and Numpy engine. It's main function is to take build Panda and Numpy structures from given tensor definition.

    Args:
        num_threads (int): The maximum number of thread the engine will use during multiprocess processing

    """
    def __init__(self, num_threads: int = None, no_logging=False):
        EngineContext.__init__(self, no_logging)
        logger.info(f'Pandas Version : {pd.__version__}')
        logger.info(f'Numpy Version : {np.__version__}')
        self._num_threads = num_threads
        self._one_hot_prefix = '__'

    @property
    def num_threads(self):
        return self._num_threads if self._num_threads is not None else int(mp.cpu_count() * 0.8)

    @property
    def one_hot_prefix(self):
        return self._one_hot_prefix

    def np_from_csv(self, target_tensor_def: Union[TensorDefinition, Tuple[TensorDefinition, ...]],
                    file: str, delimiter: chr = ',', quote: chr = "'", time_feature: Optional[Feature] = None,
                    inference: bool = True) -> TensorInstanceNumpy:
        """
        Create a Numpy Array based on a TensorDefinition by reading a file.

        Args:
            target_tensor_def: The input tensor definition. It contains all the features that need to be built. This
                function accepts either a single TensorDefinition or a Tuple of TensorDefinitions
            file: File to read. This must be a complete file path
            delimiter: The delimiter used in the file to separate columns. Default is ',' (comma)
            quote: Quote character. Default is \' (single quote)
            time_feature: Feature to use for time-based calculations. Some features need to know
                about the time such as for instance Grouper Features. Only needs to be provided if the target_tensor_def
                contains Features that need a time dimension in order to build.
            inference: (bool) Indicate if we are inferring or not. If True [COMPLETE]

        Returns:
             TensorInstanceNumpy with the Numpy arrays as defined in the target_tensor_def
        """
        # Run some basic validation
        EnginePandasValidation.val_same_feature_root_type(target_tensor_def)
        EnginePandasValidation.val_all_same_learning_category(target_tensor_def)
        EnginePandasValidation.val_no_none_learning_category(target_tensor_def)
        ttd = (target_tensor_def,) if isinstance(target_tensor_def, TensorDefinition) else target_tensor_def

        # Set up a Series processor if there are Series based Features.
        std = []
        for td in ttd:
            if any([isinstance(f, FeatureSeriesBased) for f in td.features]):
                EnginePandasValidation.val_one_feature_per_td(td)
                EnginePandasValidation.val_time_feature_needed_series(time_feature)
                std.append(td)

        if len(std) != 0:
            sp = SeriesBuilder(tuple(std), inference, self.num_threads)
        else:
            sp = None

        # Get all base feature we may need with one read into a Pandas DataFrame
        all_features = set([f for td in ttd for f in td.features])
        fts = [f for f in all_features if not isinstance(f, FeatureSeriesBased)]
        if sp is not None:
            fts.extend(sp.df_features)
            fts.append(time_feature)
        td = TensorDefinition('All_r_1', list(set(fts)))
        df = self.df_from_csv(td, file, delimiter, quote, time_feature, inference)

        # Now create the numpy lists
        npl = []
        for td in ttd:
            if any(isinstance(f, FeatureSeriesBased) for f in td.features):
                npl.append(sp.build(td, df, time_feature))
            else:
                # The Features should be in the dataframe.
                n = df[td.feature_names].to_numpy()
                td.rank = len(n.shape)
                npl.append(n)

        return TensorInstanceNumpy(ttd, tuple(npl))

    def df_from_csv(self, target_tensor_def: TensorDefinition, file: str, delimiter: chr = ',', quote: chr = "'",
                    time_feature: Optional[Feature] = None, inference: bool = True) -> pd.DataFrame:

        """
        Construct a Panda according to a tensor definition by reading a csv file from disk

        Args:
            target_tensor_def: The input tensor definition. It contains all the features that need to be built.
            file: File to read. This must be a complete file path
            delimiter: The delimiter used in the file to separate columns. Default is ',' (comma)
            quote: Quote character. Default is \' (single quote)
            time_feature: Feature to use for time-based calculations. Some features need to know
                about the time such as for instance Grouper Features. Only needs to be provided if the target_tensor_def
                contains features that need a time dimension in order to build.
            inference: (bool) Indicate if we are inferring or not. If True [COMPLETE]

        Returns:
            A Panda with the fields as defined in the tensor_def.
        """
        EnginePandasValidation.val_time_feature_needed_non_series(target_tensor_def, time_feature)
        # Check if file exists
        file_instance = pathlib.Path(file)
        if not file_instance.exists():
            raise EnginePandasException(f' path {file} does not exist or is not a file')
        logger.info(f'Building Panda for : {target_tensor_def.name} from file {file}')

        # Create a full list of feature we need to build. Including Embedded features
        need_to_build = target_tensor_def.embedded_features
        # Make sure to also build the time feature and stuff it needs
        if time_feature is not None and time_feature not in need_to_build:
            need_to_build.append(time_feature)
            need_to_build.extend(time_feature.embedded_features)

        # Make empty list of features that have already been built.
        built_features: List[Feature] = []

        # Allocate Empty Dataframe
        df = pd.DataFrame([])

        i = 1
        while len(need_to_build) > 0:
            if i > 20:
                raise EnginePandasException(
                    f'Exiting. Did more that {i} iterations trying to build {target_tensor_def.name}.' +
                    f'Potential endless loop.'
                )
            ready_to_build = [f for f in need_to_build if all(ef in built_features for ef in f.embedded_features)]
            ready_to_build = list(set(ready_to_build))
            # Start processing using the DataFrameBuilder helper class.
            dfb = DataFrameBuilder(
                ready_to_build, file, delimiter, quote, self.num_threads, self.one_hot_prefix, time_feature, inference
            )
            df = dfb.build(df)
            built_features = built_features + ready_to_build
            need_to_build = [f for f in need_to_build if f not in built_features]
            i = i+1

        # Reshape df so that it matches the original target_tensor_def
        df = self._reshape(target_tensor_def, df)
        # Don't forget to set the Tensor definition rank if in inference mode
        if not inference:
            target_tensor_def.rank = len(df.shape)
        return df

    def _reshape(self, tensor_def: TensorDefinition, df: pd.DataFrame):
        """
        Reshape function. Can be used to reshuffle the columns in a Panda. The columns will be returned according to
        the exact order as the features of the tensor definition. Columns that are not in the tensor definition as
        feature will be dropped.

        Args:
            df: Input Panda.
            tensor_def: The tensor definition according which to reshape

        Returns:
            A panda with the columns as defined in tensor_def
        """
        logger.info(f'Reshaping DataFrame to: {tensor_def.name}')
        EnginePandasValidation.val_features_in_data_frame(df, tensor_def, self.one_hot_prefix)
        col_names = []
        for feature in tensor_def.features:
            if isinstance(feature, FeatureExpander):
                col_names.extend(
                    [name for name in feature.expand_names]
                )
            else:
                col_names.append(feature.name)
        df = df[[name for name in col_names]]
        return df
