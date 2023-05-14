"""
Feature Processor for the creation of FeatureIndex features.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureIndex, FeatureTypeInteger

from .dataframebuilder import FeatureProcessor
from ..common.data import pandas_type
from ..common.exception import EnginePandasException


class FeatureIndexProcessor(FeatureProcessor[FeatureIndex]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureIndexProcessor, self).__init__(FeatureIndex, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Build a dictionary of mapping values if we are NOT in inference mode.
        if not self.inference:
            for feature in self.features:
                # Make sure to add the unknown/nan element. Make it a value that is unlikely to appear in real data.
                dct = {'*_UNK_*': 0}
                dct.update({cat: i + 1 for i, cat in enumerate(df[feature.base_feature.name].unique())})
                feature.dictionary = dct

        # Map the dictionary to the panda
        for feature in self.features:
            t = np.dtype(pandas_type(feature))
            # Check for int overflow. There could be too many values for the int type.
            if isinstance(feature.type, FeatureTypeInteger):
                self._val_int_in_range(feature, t)
            # For Panda categories we can not just fill the nans, they might not be in the categories and cause errors
            # So we must add 0 to the categories if it does not exist and then 'fill-na'.
            if df[feature.base_feature.name].dtype.name == 'category':
                if 0 not in df[feature.base_feature.name].cat.categories:
                    df[feature.base_feature.name] = df[feature.base_feature.name].cat.add_categories([0])
            df[feature.name] = df[feature.base_feature.name].map(feature.dictionary).fillna(0).astype(t)

        return df

    @staticmethod
    def _val_int_in_range(feature: FeatureIndex, d_type: np.dtype):
        """
        Method that double checks if a specific data type can hold a large enough value to store the maximum index
        value. For instance an int8 can not store more than 127 numbers.

        Args:
            feature: The index feature we that will be built
            d_type: The requested data type for the field

        Raises:
            EnginePandasException: If the integer is too small

        Returns:
            None

        """
        v_min, v_max = np.iinfo(d_type).min, np.iinfo(d_type).max
        d_s = len(feature.dictionary)
        if d_s >= v_max:
            raise EnginePandasException(f'Dictionary of {feature.name} of size {d_s} too big for type {d_type}. '
                                        + f'This will cause overflow. '
                                        + f'Please choose a data type that can hold bigger numbers')
