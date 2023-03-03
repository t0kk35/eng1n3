"""
Feature Processor for the creation of FeatureSource features.
(c) 2023 tsm
"""
import logging
import pandas as pd
from typing import List

from f3atur3s import Feature, FeatureSource, FeatureTypeTimeBased, FEATURE_TYPE_CATEGORICAL
from f3atur3s import FeatureHelper

from ..common.data import pandas_type
from .dataframebuilder import FeatureProcessor


logger = logging.getLogger(__name__)


class FeatureSourceProcessor(FeatureProcessor[FeatureSource]):
    def __init__(self, features: List[Feature], file: str, delimiter: chr, quote: chr, inference: bool):
        super(FeatureSourceProcessor, self).__init__(FeatureSource, features, inference)
        self._file = file
        self._delimiter = delimiter
        self._quote = quote

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        source_feature_names = [field.name for field in self.features]
        source_feature_types = {
            feature.name: pandas_type(feature, read=True) for feature in self.features
        }
        date_features: List[FeatureSource] = FeatureHelper.filter_feature_type(FeatureTypeTimeBased, self.features)
        # date_feature_names = [f.name for f in date_features]
        # Set up some specifics for the date/time parsing
        # date_parser = self._set_up_date_parser(date_features)
        # infer_datetime_format = True if date_parser is None else True

        df = pd.read_csv(
            self._file,
            sep=self._delimiter,
            usecols=source_feature_names,
            dtype=source_feature_types,
            quotechar=self._quote
        )

        # Reparse the date features. Make sure they become pd.DateTime
        for feature in date_features:
            df[feature.name] = pd.to_datetime(df[feature.name], format=feature.format_code)

        # Apply defaults for source data fields of type 'CATEGORICAL'
        for feature in self.features:
            if feature.default is not None:
                if feature.type == FEATURE_TYPE_CATEGORICAL:
                    if feature.default not in df[feature.name].cat.categories.values:
                        df[feature.name] = df[feature.name].cat.add_categories(feature.default)
                df[feature.name].fillna(feature.default, inplace=True)
        return df
