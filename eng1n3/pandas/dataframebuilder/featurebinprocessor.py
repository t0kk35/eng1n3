"""
Feature Processor for the creation of FeatureBin features.
(c) 2023 tsm
"""
import pandas as pd
import numpy as np
from typing import List

from f3atur3s import Feature, FeatureBin

from .processor import FeatureProcessor
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
            cut = pd.cut(df[feature.base_feature.name], bins=bins, labels=labels)
            df[feature.name] = cut.cat.add_categories(0).fillna(0)

        return df
