"""
Feature Processor for the creation of FeatureOneHot features.
(c) 2023 tsm
"""
import numpy as np
import pandas as pd
from typing import List

from f3atur3s import Feature, FeatureOneHot
from ..common.exception import EnginePandasException
from ..dataframebuilder.dataframebuilder import FeatureProcessor


class FeatureOneHotProcessor(FeatureProcessor[FeatureOneHot]):
    def __init__(self, features: List[Feature], one_hot_prefix: str, inference: bool):
        super(FeatureOneHotProcessor, self).__init__(FeatureOneHot, features, inference)
        self._one_hot_prefix = one_hot_prefix

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        original_df = df[[f.base_feature.name for f in self.features]]

        if not self.inference:
            # Use pandas function to get the one-hot features. Set the 'expand names' inference attribute
            columns = [feature.base_feature.name for feature in self.features]
            df = pd.get_dummies(df, prefix_sep=self._one_hot_prefix, columns=columns)
            for oh in self.features:
                oh.expand_names = [c for c in df.columns if c.startswith(oh.base_feature.name + self._one_hot_prefix)]
                if len(oh.expand_names) == 0:
                    raise EnginePandasException(
                        f'Did not create any field for OneHot feature {oh.name}. This most is most likely because the' +
                        f' base feature {oh.base_feature.name} is always empty or all Nan'
                    )
        else:
            # During inference the values might be different. Need to make sure the number of columns matches
            # the training values. Values that were not seen during training will be removed.
            # Values that were seen during training but not at inference need to be added with all zeros
            columns = [feature.base_feature.name for feature in self._features]
            df = pd.get_dummies(df, prefix_sep=self._one_hot_prefix, columns=columns)
            # Add features seen at non-inference (training), but not at inference
            n_defined = []
            for f in self.features:
                defined = [col for col in df.columns if col.startswith(f.base_feature.name + self._one_hot_prefix)]
                x = [n for n in f.expand_names if n not in defined]
                n_defined.extend(x)
            if len(n_defined) > 0:
                kwargs = {nd: np.zeros((len(df),), dtype='int8') for nd in n_defined}
                df = df.assign(**kwargs)
            # Remove features not seen at non-inference (training) but seen at inference
            n_defined = []
            for f in self.features:
                x = [col for col in df.columns
                     if col.startswith(f.base_feature.name + self._one_hot_prefix)
                     and col not in f.expand_names]
                n_defined.extend(x)
            if len(n_defined) > 0:
                df = df.drop(n_defined, axis=1)

        # Pandas will have removed the original features when 'get_dummies' is called
        # Add back the original features. We might need them later
        return pd.concat([original_df, df], axis=1)
