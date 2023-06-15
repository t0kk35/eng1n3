"""
Unit Tests for PandasNumpy Engine, specifically the FeatureDateTimeFormat usage
(c) 2023 tsm
"""
import unittest

import numpy as np
import pandas as pd

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


class TestFeatureDateTimeFormat(unittest.TestCase):
    def test_read_base_non_inference(self):
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fmt_name = 'date_fmt'
        # Extract the week day as decimal
        fmt = '%w'
        ff = ft.FeatureDateTimeFormat(fmt_name, ft.FEATURE_TYPE_INT_8, fd, fmt)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fd])
            df = e.df_from_csv(td, file, inference=False)
            dts = df['Date'].dt.strftime(fmt).astype('int8')
            td2 = ft.TensorDefinition('Derived', [ff])
            df = e.df_from_csv(td2, file, inference=False)
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertTrue(pd.DataFrame.equals(dts, df[fmt_name]), f'DataFrames not equal {dts}, {df[fmt_name]}')
            self.assertEqual(ff.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(df.columns[0], fmt_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'int8', f'Expecting a "int8" data type')
            self.assertEqual(ff.inference_ready, True, f'DateTimeFormat feature should be ready for inference')
            self.assertEqual(len(ff.embedded_features), 1, f'Expecting one embedded feature')
            self.assertListEqual(ff.embedded_features, [fd], f'Expecting fa to be the embedded feature')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(td2.inference_ready, True, f'TensorDefinition should be ready for inference')
            self.assertListEqual(td2.categorical_features(), [ff], f'Expanded Feature not correct')
            self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embedded feats {len(td2.embedded_features)}')
            self.assertListEqual(
                sorted(td2.embedded_features, key=lambda x: x.name),
                sorted([fd, ff], key=lambda x: x.name), f'Embedded features should be fd and ff'
            )


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        fmt = '%w'
        fb = ft.FeatureDateTimeFormat('BinAmount', ft.FEATURE_TYPE_INT_16, fa, fmt)
        td1 = ft.TensorDefinition('TestNP1', [fb])
        td2 = ft.TensorDefinition('TestNP2', [fb])
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td1, file, inference=False)
            n = e.np_from_csv(td2, file, inference=False)
        self.assertEqual(type(n), en.TensorInstanceNumpy, f'Did not get TensorInstanceNumpy. But {type(n)}')
        self.assertEqual(len(n.numpy_lists), 1, f'Expected only one list. Got {len(n.numpy_lists)}')
        self.assertEqual(len(n), len(df), f'Lengths not equal {len(df)}, {len(n)}')
        self.assertTrue(np.all(np.equal(df.to_numpy(), n.numpy_lists[0])), f'from np not OK. {df}, {n.numpy_lists[0]}')
        self.assertEqual(td1.rank, td2.rank, f'Expecting the same td rank {td1.rank} {td2.rank}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
