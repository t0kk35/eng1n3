"""
Unit Tests for PandasNumpy Engine, specifically the FeatureDateTimeWave usage
(c) 2023 tms
"""
import unittest

import numpy as np
import pandas as pd

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


class TestFeatureDateTimeWave(unittest.TestCase):
    def test_read_base(self):
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fw_name = 'date_fw'
        # Extract the week day as decimal
        fmt = '%w'
        period = 7
        frequencies = 3
        fw = ft.FeatureDateTimeWave(fw_name, ft.FEATURE_TYPE_FLOAT, fd, fmt, period, frequencies)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('Wave', [fw])
            df = e.df_from_csv(td, file, inference=False)
            self.assertEqual(len(df.columns), frequencies * 2, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(fw.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertListEqual(list(df.columns),
                                 [f'{fw.name}{fw.delimiter}{w}{fw.delimiter}{f}'
                                  for f in range(fw.frequencies)
                                  for w in ('sin', 'cos')], f'Column name incorrect. Got {df.columns}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'float64', f'Expecting a "float64" data type')
            self.assertEqual(fw.inference_ready, True, f'DateTimeFormat feature should be ready for inference')
            self.assertEqual(len(fw.embedded_features), 1, f'Expecting one embedded feature')
            self.assertListEqual(fw.embedded_features, [fd], f'Expecting fd to be the embedded feature')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(td.inference_ready, True, f'TensorDefinition should be ready for inference')
            self.assertListEqual(td.continuous_features(), [fw], f'Expanded Feature not correct')
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embedded feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fd, fw], key=lambda x: x.name), f'Embedded features should be fd and fw'
            )

    def test_build_2(self):
        fs = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fw_name = 'date_fw'
        fd_name = 'date_fd'
        # Extract the week day as decimal
        fmt_w = '%w'
        fmt_d = '%d'
        period = 7
        frequencies = 1
        fw = ft.FeatureDateTimeWave(fw_name, ft.FEATURE_TYPE_FLOAT, fs, fmt_w, period, frequencies)
        fd = ft.FeatureDateTimeWave(fd_name, ft.FEATURE_TYPE_FLOAT, fs, fmt_d, period, frequencies)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('Wave', [fw, fd])
            df = e.df_from_csv(td, file, inference=False)
            self.assertEqual(len(df.columns), frequencies * 4, f'Should have gotten 4 columns. Got {len(df.columns)}')
            self.assertEqual(fw.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertEqual(fd.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertListEqual(list(df.columns),
                                 [f'{fw.name}{fw.delimiter}{w}{fw.delimiter}{f}'
                                  for f in range(fw.frequencies)
                                  for w in ('sin', 'cos')] +
                                 [f'{fd.name}{fd.delimiter}{w}{fd.delimiter}{f}'
                                  for f in range(fd.frequencies)
                                  for w in ('sin', 'cos')]
                                 , f'Column name incorrect. Got {df.columns}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'float64', f'Expecting a "float64" data type')
            self.assertEqual(fw.inference_ready, True, f'DateTimeFormat feature should be ready for inference')
            self.assertEqual(fd.inference_ready, True, f'DateTimeFormat feature should be ready for inference')
            self.assertEqual(len(fw.embedded_features), 1, f'Expecting one embedded feature')
            self.assertEqual(len(fd.embedded_features), 1, f'Expecting one embedded feature')
            self.assertListEqual(fw.embedded_features, [fs], f'Expecting fd to be the embedded feature')
            self.assertListEqual(fd.embedded_features, [fs], f'Expecting fd to be the embedded feature')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(td.inference_ready, True, f'TensorDefinition should be ready for inference')
            self.assertListEqual(td.continuous_features(), [fw, fd], f'Expanded Feature not correct')
            self.assertEqual(len(td.embedded_features), 3, f'Expecting 2 embedded feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fs, fd, fw], key=lambda x: x.name), f'Embedded features should be fd and fw'
            )


def main():
    unittest.main()


if __name__ == '__main__':
    main()