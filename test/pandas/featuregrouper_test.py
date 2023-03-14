"""
Unit Tests for PandasNumpy Engine, specifically the FeatureGrouper usage
(c) 2020 tsm
"""
import unittest
import os
import pandas as pd
from datetime import datetime, timedelta
from statistics import stdev
from typing import List

import numpy as np

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


def fn_not_one(x: float) -> bool:
    return x != 1.0


def fn_one(x: float) -> bool:
    return x == 1.0


class TestGrouperFeature(unittest.TestCase):
    def test_grouped_bad_no_time_feature(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fx = ft.FeatureExpression('DateDerived', ft.FEATURE_TYPE_DATE, fn_one, [fd])
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        fg = ft.FeatureGrouper(
            '2_day_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM)
        td2 = ft.TensorDefinition('Derived', [fd, fr, fa, ff, fg])
        tdx = ft.TensorDefinition('Derived', [fx, fr, fa, ff, fg])
        with en.EnginePandas(num_threads=1) as e:
            # No time feature parameter is bad, needs to be provided
            with self.assertRaises(en.EnginePandasException):
                _ = e.df_from_csv(td2, file, inference=False)
            # No time feature is bad. if it is derived also; i.e. embedded.
            with self.assertRaises(en.EnginePandasException):
                _ = e.df_from_csv(tdx, file, inference=False)
            # Time Feature not of datetime type is also bad
            with self.assertRaises(en.EnginePandasException):
                _ = e.df_from_csv(td2, file, time_feature=fa, inference=False)

    def test_grouped_bad_base_not_float(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        fg = ft.FeatureGrouper(
            '2_day_sum', ft.FEATURE_TYPE_FLOAT_32, ff, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM)
        td = ft.TensorDefinition('Derived', [fd, fr, fa, ff, fg])
        with en.EnginePandas(num_threads=1) as e:
            with self.assertRaises(en.EnginePandasException):
                _ = e.df_from_csv(td, file, time_feature=fa, inference=False)

    def test_grouped_single_window_all_aggregates(self):
        # Base test. Create single aggregate daily sum on card
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        time_window = 2

        feature_def = [
            ('2_day_sum', ft.AGGREGATOR_SUM, sum),
            ('2_day_cnt', ft.AGGREGATOR_COUNT, len),
            ('2_day_avg', ft.AGGREGATOR_AVG, lambda x: sum(x) / len(x)),
            ('2_day_min', ft.AGGREGATOR_MIN, min),
            ('2_day_max', ft.AGGREGATOR_MAX, max),
            ('2_day_std', ft.AGGREGATOR_STDDEV, lambda x: 0 if len(x) == 1 else stdev(x))
        ]
        group_features = [
            ft.FeatureGrouper(name, ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, time_window, agg)
            for name, agg, _ in feature_def
        ]
        features: List[ft.Feature] = [
            fd, fr, fa, ff
        ]
        features.extend(group_features)
        td = ft.TensorDefinition('Features', features)
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td, file, inference=False, time_feature=fd)

        # Check that all GrouperFeatures have been created.
        for grouper_name, _, _ in feature_def:
            self.assertIn(grouper_name, df.columns, f'The aggregated feature {grouper_name} not found in Pandas')

        # Check that all types are correct. They should all have been created as float 32
        for f in group_features:
            self.assertEqual(df[f.name].dtype.name, 'float32', f'Type not FP32 for {df[f.name]}, {df[f.name].dtype}')

        # Check the aggregates. Iterate over the Card-id (the key) and check each aggregate
        for c in df[fr.name].unique():
            prev_dt = None
            amounts = []
            for _, row in df.iterrows():
                if row[fr.name] == c:
                    if prev_dt is None:
                        amounts.append(row[fa.name])
                        for grouper_name, _, list_fn in feature_def:
                            amt = list_fn(amounts)
                            g_amt = row[grouper_name]
                            self.assertAlmostEqual(amt, g_amt, 6, f'Expected {grouper_name} to be same as the amount')
                    elif row[fd.name] >= (prev_dt + timedelta(days=time_window)):
                        amounts = [row[fa.name]]
                        for grouper_name, _, list_fn in feature_def:
                            amt = list_fn(amounts)
                            g_amt = row[grouper_name]
                            self.assertAlmostEqual(amt, g_amt, 6, f'Expected {grouper_name} to be {g_amt}')
                    else:
                        amounts.append(row[fa.name])
                        for grouper_name, _, list_fn in feature_def:
                            amt = list_fn(amounts)
                            g_amt = row[grouper_name]
                            # Do almost equal. There's a slight difference in stddev between pandas and the stats impl
                            self.assertAlmostEqual(amt, g_amt, 6, f'Expected {grouper_name} to be {g_amt}')
                    prev_dt = row[fd.name]

    def test_grouped_multiple_groups(self):
        # Base test. Create single all aggregates on 2 groups
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fga_1 = ft.FeatureGrouper(
            'card_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM
        )
        fga_2 = ft.FeatureGrouper(
            'card_2d_count', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_COUNT
        )
        fga_3 = ft.FeatureGrouper(
            'card_2d_avg', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_AVG
        )
        fga_4 = ft.FeatureGrouper(
            'card_2d_std', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_STDDEV
        )
        fga_5 = ft.FeatureGrouper(
            'card_2d_min', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MIN
        )
        fga_6 = ft.FeatureGrouper(
            'card_2d_max', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MAX
        )
        fgc_1 = ft.FeatureGrouper(
            'country_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM
        )
        fgc_2 = ft.FeatureGrouper(
            'country_2d_count', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_COUNT
        )
        fgc_3 = ft.FeatureGrouper(
            'country_2d_avg', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_AVG
        )
        fgc_4 = ft.FeatureGrouper(
            'country_2d_std', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_STDDEV
        )
        fgc_5 = ft.FeatureGrouper(
            'country_2d_min', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MIN
        )
        fgc_6 = ft.FeatureGrouper(
            'country_2d_max', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MAX
        )

        td2 = ft.TensorDefinition('Derived', [fga_1, fga_2, fga_3, fga_4, fga_5, fga_6])
        td3 = ft.TensorDefinition('Derived', [fgc_1, fgc_2, fgc_3, fgc_4, fgc_5, fgc_6])
        td4 = ft.TensorDefinition('Derived', [fga_1, fga_2, fga_3, fga_4, fga_5, fga_6,
                                              fgc_1, fgc_2, fgc_3, fgc_4, fgc_5, fgc_6])
        with en.EnginePandas(num_threads=1) as e:
            df_card = e.df_from_csv(td2, file, inference=False, time_feature=fd)
            df_country = e.df_from_csv(td3, file, inference=False, time_feature=fd)
            df_comb = e.df_from_csv(td4, file, inference=False, time_feature=fd)
            # The resulting df_comb (with 2 groups) should be the same and doing each individual and concatenating
            self.assertTrue(df_comb.equals(pd.concat([df_card, df_country], axis=1)), f'Concatenate dataframe not ==')

    def test_filter(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fl = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        f_not_one = ft.FeatureFilter('Not_Fraud', ft.FEATURE_TYPE_BOOL, fn_not_one, [fl])
        f_is_one = ft.FeatureFilter('Is_Fraud', ft.FEATURE_TYPE_BOOL, fn_one, [fl])
        fg_not_one = ft.FeatureGrouper(
            'card_not_one', ft.FEATURE_TYPE_FLOAT_32, fa, fr, f_not_one, ft.TIME_PERIOD_DAY, 1, ft.AGGREGATOR_SUM
        )
        fg_is_one = ft.FeatureGrouper(
            'card_one', ft.FEATURE_TYPE_FLOAT_32, fa, fr, f_is_one, ft.TIME_PERIOD_DAY, 1, ft.AGGREGATOR_SUM
        )
        fg_no_filter = ft.FeatureGrouper(
            'card_no_filter', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 1, ft.AGGREGATOR_SUM
        )
        td = ft.TensorDefinition('Derived', [fd, fr, fc, fa, f_not_one, f_is_one, fg_not_one, fg_is_one, fg_no_filter])
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td, file, inference=False, time_feature=fd)
        # We are using a 1-day window so the filtered records on (fraud == 1) added to (fraud == 2)
        # should be the same a no filter
        self.assertTrue((df['card_no_filter'].equals(df['card_not_one'] + df['card_one'])))


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fg = ft.FeatureGrouper(
            '2_day_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM)
        td = ft.TensorDefinition('TestNP', [fg])
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td, file, inference=False, time_feature=fd)
            n = e.np_from_csv(td, file, inference=False, time_feature=fd)
        self.assertEqual(type(n), en.TensorInstanceNumpy, f'Did not get TensorInstanceNumpy. But {type(n)}')
        self.assertEqual(len(n.numpy_lists), 1, f'Expected only one list. Got {len(n.numpy_lists)}')
        self.assertEqual(len(n), len(df), f'Lengths not equal {len(df)}, {len(n)}')
        self.assertTrue(np.all(np.equal(df.to_numpy(), n.numpy_lists[0])), f'from np not OK. {df}, {n.numpy_lists[0]}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
