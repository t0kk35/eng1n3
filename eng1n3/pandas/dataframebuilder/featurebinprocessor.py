"""
Feature Processor for the creation of FeatureBin features.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureBin

from .featureprocessor import FeatureProcessor
from ..common.data import pandas_type


class FeatureBinProcessor(FeatureProcessor[FeatureBin]):
    def __init__(self, features: List[Feature], inference: bool):
        super(FeatureBinProcessor, self).__init__(FeatureBin, features, inference)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add the binning features
        for feature in self.features:
            if not self.inference:
                # Geometric space can not start for 0
                mx = df[feature.base_feature.name].max()
                if feature.scale_type == 'geometric':
                    mn = max(df[feature.base_feature.name].min(), 1e-1)
                    bins = np.geomspace(mn, mx, feature.number_of_bins)
                else:
                    mn = df[feature.base_feature.name].min()
                    bins = np.linspace(mn, mx, feature.number_of_bins)
                # Set last bin to max possible value. Otherwise, unseen values above the biggest bin go to 0.
                bins[-1] = np.finfo(bins.dtype).max
                # Set inference attributes
                feature.bins = list(bins)
            bins = np.array(feature.bins)
            t = np.dtype(pandas_type(feature))
            labels = np.array(feature.range).astype(np.dtype(t))
            # Set all values that are smaller than the min bin to the min bin. Otherwise, both amounts lower
            # than min bin and missing value will become NaN and will be indistinguishable
            min_adj = np.where(df[feature.base_feature.name] < bins[0], bins[0], df[feature.base_feature.name])
            cut = pd.cut(min_adj, bins=bins, labels=labels, include_lowest=True)
            # Add 1 so the 0 bin is reserved for unknown/unseen. Cut will return -1 as code for NaN values.
            df[feature.name] = cut.codes.astype(np.dtype(t)) + 1
            # If this point in time we have a 0 index, then the original value was -1, meaning the value was smaller
            # than the smallest bin. We're going to rid of those, and assign then to the smallest bin,
            # we really want 0 index to mean; 'missing'.
            # Now all Nan (missing) become the 0 index.
            # df[feature.name] = df[feature.name].fillna(0)

        return df
