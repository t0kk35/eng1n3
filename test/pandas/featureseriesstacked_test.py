"""
Unit Tests for PandasNumpy Engine, specifically the FeatureRatio usage
(c) 2023 tsm
"""
import unittest

import numpy as np

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


class TestBaseCreate(unittest.TestCase):
    def test_create(self):
        series_len = 2
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        fci = ft.FeatureIndex('Country_I', ft.FEATURE_TYPE_INT_16, fc)
        fmi = ft.FeatureIndex('MCC_I', ft.FEATURE_TYPE_INT_16, fm)
        series_ft = [fmi, fci]
        fs = ft.FeatureSeriesStacked('Stacked_Categorical', ft.FEATURE_TYPE_INT_16, series_ft, series_len, fr)
        td0 = ft.TensorDefinition('Key', [fr])
        td1 = ft.TensorDefinition('Original', series_ft)
        td2 = ft.TensorDefinition('Derived', [fs])
        with en.EnginePandas(num_threads=1) as e:
            k = e.df_from_csv(td0, file, inference=False)
            n = e.np_from_csv(td1, file, inference=False)
            s = e.np_from_csv(td2, file, time_feature=fd, inference=False)
        self.assertTrue(isinstance(s, en.TensorInstanceNumpy), f'Expecting a TensorNumpyInstance, got type {type(s)}')
        self.assertEqual(len(s.numpy_lists), 1, f'Expecting length one, got {len(s)}')
        self.assertEqual(s.numpy_lists[0].dtype, 'int16', f'Should have returned the type of the input features')
        self.assertEqual(s.numpy_lists[0].shape[1], series_len, f'time dimension not correct. Expecting {series_len}')
        self.assertEqual(s.numpy_lists[0].shape[2], len(series_ft), f'Feature dimension not correct. {len(series_ft)}')
        for key in k['Card'].unique():
            mask = k['Card'] == key
            c_n = n.numpy_lists[0][mask]
            c_s = s.numpy_lists[0][mask]
            for i in range(c_s.shape[0]):
                if i < series_len:
                    # Check pre-padding w/zeros
                    self.assertTrue(np.array_equal(
                        c_s[i][0:series_len-i-1],
                        np.zeros((series_len-i-1, len(series_ft)))
                    ))
                    self.assertTrue(np.array_equal(
                        c_s[i][series_len-i-1:],
                        c_n[0:i+1]
                    ), f'Data after padding not correct {c_s[i][series_len-i-1:]} {c_n[0:i+1]}')
                else:
                    self.assertTrue(np.array_equal(
                        c_s[i],
                        c_n[i - series_len + 1:i + 1]
                    ), f'Data after padding not correct {c_s[i]} {c_n[series_len-i-1:i]}')
        self.assertTrue(td2.inference_ready, f'TensorDefinition should have been ready for inference')
        self.assertEqual(td2.rank, 3, f'Rank of TensorDefinition should have been 3')

    def test_multiple_td(self):
        series_len = 3
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        fci = ft.FeatureIndex('Country_I', ft.FEATURE_TYPE_INT_16, fc)
        fmi = ft.FeatureIndex('MCC_I', ft.FEATURE_TYPE_INT_16, fm)
        fs1 = ft.FeatureSeriesStacked('Stacked_MCC', ft.FEATURE_TYPE_INT_16, [fmi], series_len, fr)
        fs2 = ft.FeatureSeriesStacked('Stacked_Country', ft.FEATURE_TYPE_INT_16, [fci], series_len, fr)
        td0 = ft.TensorDefinition('Key', [fr])
        td1 = ft.TensorDefinition('Original', [fmi, fci])
        td2 = ft.TensorDefinition('Series1', [fs1])
        td3 = ft.TensorDefinition('Series2', [fs2])
        with en.EnginePandas(num_threads=1) as e:
            k = e.df_from_csv(td0, file, inference=False)
            n = e.np_from_csv(td1, file, inference=False)
            s = e.np_from_csv((td2, td3), file, time_feature=fd, inference=False)
        self.assertTrue(isinstance(s, en.TensorInstanceNumpy), f'Expecting a TensorNumpyInstance, got type {type(s)}')
        self.assertEqual(len(s.numpy_lists), 2, f'Expecting 2 numpy lists')
        for npl in s.numpy_lists:
            self.assertEqual(npl.dtype, 'int16', f'Should have returned the type of the input features')
            self.assertEqual(npl.shape[1], series_len, f'time dimension not correct. Expecting {series_len}')
            self.assertEqual(npl.shape[2], 1, f'Feature dimension not correct. Should have been 1')
        for key in k['Card'].unique():
            mask = k['Card'] == key
            c_n_0 = n.numpy_lists[0][:, 0][mask]
            c_s_0 = s.numpy_lists[0][mask]
            c_n_1 = n.numpy_lists[0][:, 1][mask]
            c_s_1 = s.numpy_lists[1][mask]
            for i in range(c_s_0.shape[0]):
                if i < series_len:
                    # Check pre-padding w/zeros
                    self.assertTrue(np.array_equal(
                        c_s_0[i][0:series_len-i-1],
                        np.zeros((series_len-i-1, 1))
                    ))
                    self.assertTrue(np.array_equal(
                        c_s_1[i][0:series_len-i-1],
                        np.zeros((series_len-i-1, 1))
                    ))
                    self.assertTrue(np.array_equal(
                        np.squeeze(c_s_0[i][series_len-i-1:], axis=1),
                        c_n_0[0:i+1]
                    ), f'Data after padding not correct {i} {c_s_0[i][series_len-i-1:]} {c_n_0[0:i+1]}')
                    self.assertTrue(np.array_equal(
                        np.squeeze(c_s_1[i][series_len-i-1:], axis=1),
                        c_n_1[0:i+1]
                    ), f'Data after padding not correct {c_s_0[i][series_len-i-1:]} {c_n_0[0:i+1]}')
                else:
                    self.assertTrue(np.array_equal(
                        c_s_0[i],
                        c_n_0[i - series_len + 1:i + 1]
                    ), f'Data after padding not correct {c_s_0[i]} {c_n[series_len-i-1:i]}')
                    self.assertTrue(np.array_equal(
                        c_s_1[i],
                        c_n_1[i - series_len + 1:i + 1]
                    ), f'Data after padding not correct {c_s_1[i]} {c_n_1[series_len-i-1:i]}')


class TestCombined(unittest.TestCase):
    def test_combined_series_non_series(self):
        series_len = 3
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        fci = ft.FeatureIndex('Country_I', ft.FEATURE_TYPE_INT_16, fc)
        fmi = ft.FeatureIndex('MCC_I', ft.FEATURE_TYPE_INT_16, fm)
        fs1 = ft.FeatureSeriesStacked('Series', ft.FEATURE_TYPE_INT_16, [fmi, fci], series_len, fr)
        td1 = ft.TensorDefinition('No-Series', [fmi, fci])
        td2 = ft.TensorDefinition('Series-1', [fs1])
        with en.EnginePandas(num_threads=1) as e:
            ns = e.np_from_csv(td1, file, inference=False)
            s = e.np_from_csv(td2, file, time_feature=fd, inference=False)
            c = e.np_from_csv((td1, td2), file, time_feature=fd, inference=False)
            self.assertTrue(isinstance(ns, en.TensorInstanceNumpy), 'Expecting a TensorInstanceNumpy')
            self.assertTrue(isinstance(s, en.TensorInstanceNumpy), 'Expecting a TensorInstanceNumpy')
            self.assertTrue(isinstance(c, en.TensorInstanceNumpy), 'Expecting a TensorInstanceNumpy')
            self.assertEqual(len(ns.numpy_lists), 1, f'Expecting 1 numpy list')
            self.assertEqual(len(s.numpy_lists), 1, f'Expecting 1 numpy list')
            self.assertEqual(len(c.numpy_lists), 2, f'Expecting 2 numpy lists')
            self.assertTrue(np.array_equal(c.numpy_lists[0], ns.numpy_lists[0]))
            self.assertTrue(np.array_equal(c.numpy_lists[1], s.numpy_lists[0]))


class TestFail(unittest.TestCase):
    def test_only_one_series_feature_per_td(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        fci = ft.FeatureIndex('Country_I', ft.FEATURE_TYPE_INT_16, fc)
        fmi = ft.FeatureIndex('MCC_I', ft.FEATURE_TYPE_INT_16, fm)
        fs1 = ft.FeatureSeriesStacked('Stacked_Country', ft.FEATURE_TYPE_INT_16, [fci], 3, fr)
        fs2 = ft.FeatureSeriesStacked('Stacked_MCC', ft.FEATURE_TYPE_INT_16, [fmi], 3, fr)
        td = ft.TensorDefinition('Fail', [fs1, fs2])
        with en.EnginePandas(num_threads=1) as e:
            with self.assertRaises(en.EnginePandasException):
                _ = e.np_from_csv(td, file, time_feature=fd, inference=False)

    def test_time_feature_needed(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        fci = ft.FeatureIndex('Country_I', ft.FEATURE_TYPE_INT_16, fc)
        fmi = ft.FeatureIndex('MCC_I', ft.FEATURE_TYPE_INT_16, fm)
        fs1 = ft.FeatureSeriesStacked('Stacked_Country', ft.FEATURE_TYPE_INT_16, [fci], 3, fr)
        fs2 = ft.FeatureSeriesStacked('Stacked_MCC', ft.FEATURE_TYPE_INT_16, [fmi], 3, fr)
        td = ft.TensorDefinition('Fail', [fs1, fs2])
        with en.EnginePandas(num_threads=1) as e:
            with self.assertRaises(en.EnginePandasException):
                _ = e.np_from_csv(td, file, inference=False)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
