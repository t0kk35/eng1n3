"""
Feature Processor for the creation of FeatureSource features.
(c) 2023 tsm
"""
import logging
import pandas as pd
import datetime as dt
from functools import partial
from typing import List, Optional

from f3atur3s import Feature, FeatureSource, FeatureTypeTimeBased, FEATURE_TYPE_CATEGORICAL
from f3atur3s import FeatureHelper

from ..common.data import pandas_type
from ..helpers.validation import EnginePandasValidation
from eng1n3.pandas.dataframebuilder import FeatureProcessor


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
        date_feature_names = [f.name for f in date_features]
        # Set up some specifics for the date/time parsing
        date_parser = self._set_up_date_parser(date_features)
        infer_datetime_format = True if date_parser is None else True

        df = pd.read_csv(
            self._file,
            sep=self._delimiter,
            usecols=source_feature_names,
            dtype=source_feature_types,
            quotechar=self._quote,
            parse_dates=date_feature_names,
            date_parser=date_parser,
            infer_datetime_format=infer_datetime_format
        )

        # Apply defaults for source data fields of type 'CATEGORICAL'
        for feature in self.features:
            if feature.default is not None:
                if feature.type == FEATURE_TYPE_CATEGORICAL:
                    if feature.default not in df[feature.name].cat.categories.values:
                        df[feature.name] = df[feature.name].cat.add_categories(feature.default)
                df[feature.name].fillna(feature.default, inplace=True)
        return df

    def _set_up_date_parser(self, date_features: List[FeatureSource]) -> Optional[partial]:
        """
        Helper function to which sets-up a data parser for a specific format. The date parser is used by the pandas
        read_csv function.

        Args:
             date_features: (List[FeaturesSource]) the Features of type date which need to be read

        Returns:
            A function (the date parser) or none if there were no explicitly defined formats
        """
        if len(date_features) != 0:
            format_codes = list(set([d.format_code for d in date_features]))
            EnginePandasValidation.val_single_date_format_code(format_codes)
            return partial(self._parse_dates, format_code=format_codes[0])
        else:
            return None

    @staticmethod
    def _parse_dates(dates: List[str], format_code: str) -> List[dt.datetime]:
        """
        Helper function to parse datetime structures from strings

        Args:
            dates: A list of dates to parse (as string)
            format_code: The format code to apply

        Returns:
             List of datetime type values
        """
        return [dt.datetime.strptime(d, format_code) for d in dates]
