"""
Unit Tests for PandasNumpy Engine, specifically the FeatureBin usage
(c) 2020 d373c7
"""
import unittest
import os
import pandas as pd
from datetime import datetime, timedelta

import numpy as np

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


def test_expr(x: float) -> float:
    return x + 1


def zero_expr() -> float:
    return 0.0


def test_date_expr(x: int) -> datetime:
    y = pd.to_datetime(x)
    y = y + timedelta(days=1)
    return y


class TestFeatureExpression(unittest.TestCase):
    """Test cases for Expression Feature"""
    def test_expr_lambda_float(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fe_name = 'AddAmount'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fe = ft.FeatureExpression(fe_name, ft.FEATURE_TYPE_FLOAT_32, lambda x: x+1, [fa])
        td1 = ft.TensorDefinition('base', [fa])
        td2 = ft.TensorDefinition('derived', [fe])
        with en.EnginePandas(num_threads=1) as e:
            df1 = e.from_csv(td1, file, inference=False)
            df2 = e.from_csv(td2, file, inference=False)
        df1['Amount'] = df1['Amount'] + 1
        self.assertTrue(df1['Amount'].equals(df2['AddAmount']), f'Amounts should have been equal')
        self.assertEqual(df2.columns[0], fe_name, f'Column name incorrect. Got {df2.columns[0]}')
        self.assertEqual(df2.iloc[:, 0].dtype.name, 'float32', f'Expecting a object data type')
        self.assertEqual(fe.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
        self.assertEqual(fe.inference_ready, True, f'Expression feature should be ready for inference')
        self.assertEqual(len(fe.embedded_features), 1, f'Expecting 1 embedded ft. Got {len(fe.embedded_features)}')
        self.assertEqual(fe.embedded_features[0], fa, 'Embedded feature should be the original source feature')
        self.assertEqual(td2.inference_ready, True, f'Tensor should still be ready for inference')
        self.assertEqual(td2.rank, 2, f'This should have been a rank 2 tensor. Got {td2.rank}')
        self.assertListEqual(td2.continuous_features(), [fe], f'Expanded Feature not correct')
        self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embedded feats {len(td2.embedded_features)}')
        self.assertListEqual(
            sorted(td2.embedded_features, key=lambda x: x.name),
            sorted([fa, fe], key=lambda x: x.name), f'Embedded features should be fa and fe'
        )

    def test_expr_non_lambda_float(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fe_name = 'AddAmount'
        fe = ft.FeatureExpression(fe_name, ft.FEATURE_TYPE_FLOAT_32, test_expr, [fa])
        td1 = ft.TensorDefinition('base', [fa])
        td2 = ft.TensorDefinition('derived', [fe])
        with en.EnginePandas(num_threads=1) as e:
            df1 = e.from_csv(td1, file, inference=False)
            df2 = e.from_csv(td2, file, inference=False)
        df1['Amount'] = df1['Amount'] + 1
        self.assertTrue(df1['Amount'].equals(df2['AddAmount']), f'Amounts should have been equal')
        self.assertEqual(df2.columns[0], fe_name, f'Column name incorrect. Got {df1.columns[0]}')
        self.assertEqual(df2.iloc[:, 0].dtype.name, 'float32', f'Expecting a object data type')
        self.assertEqual(fe.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting None LC')
        self.assertEqual(fe.inference_ready, True, f'Expression feature should be ready for inference')
        self.assertEqual(len(fe.embedded_features), 1, f'Expecting 1 embedded ft. Got {len(fe.embedded_features)}')
        self.assertEqual(fe.embedded_features[0], fa, 'Embedded feature should be the original source feature')
        self.assertEqual(td2.inference_ready, True, f'Tensor should still be ready for inference')
        self.assertEqual(td2.rank, 2, f'This should have been a rank 2 tensor. Got {td2.rank}')
        self.assertListEqual(td2.continuous_features(), [fe], f'Expanded Feature not correct')
        self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embedded feats {len(td2.embedded_features)}')
        self.assertListEqual(
            sorted(td2.embedded_features, key=lambda x: x.name),
            sorted([fa, fe], key=lambda x: x.name), f'Embedded features should be fa and fe'
        )

    def test_expr_non_lambda_date(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fe_name = 'Add1ToDate'
        fe = ft.FeatureExpression(fe_name, ft.FEATURE_TYPE_DATE, test_date_expr, [fd])
        td1 = ft.TensorDefinition('base', [fd])
        td2 = ft.TensorDefinition('derived', [fe])
        with en.EnginePandas(num_threads=1) as e:
            df1 = e.from_csv(td1, file, inference=False)
            df2 = e.from_csv(td2, file, inference=False)
            df1['Date'] = df1['Date'] + timedelta(days=1)
            self.assertTrue(df1['Date'].equals(df2[fe_name]), f'Dates should have been equal')
        self.assertEqual(df2.columns[0], fe_name, f'Column name incorrect. Got {df1.columns[0]}')
        self.assertEqual(df2.iloc[:, 0].dtype.name, 'datetime64[ns]', f'Expecting a object data type')
        self.assertEqual(fe.learning_category, ft.LEARNING_CATEGORY_NONE, f'Expecting None LC')
        self.assertEqual(fe.inference_ready, True, f'Expression feature should be ready for inference')
        self.assertEqual(len(fe.embedded_features), 1, f'Expecting 1 embedded ft. Got {len(fe.embedded_features)}')
        self.assertEqual(fe.embedded_features[0], fd, 'Embedded feature should be the original source feature')
        self.assertEqual(td2.inference_ready, True, f'Tensor should still be ready for inference')
        self.assertEqual(td2.rank, 2, f'This should have been a rank 2 tensor. Got {td2.rank}')
        self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embedded feats {len(td2.embedded_features)}')
        self.assertListEqual(
            sorted(td2.embedded_features, key=lambda x: x.name),
            sorted([fe, fd], key=lambda x: x.name), f'Embedded features should be fe and fd'
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
