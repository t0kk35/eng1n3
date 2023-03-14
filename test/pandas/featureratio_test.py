"""
Unit Tests for PandasNumpy Engine, specifically the FeatureRatio usage
(c) 2023 d373c7
"""
import unittest
import os
import numpy as np
import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


def test_expr(i: float) -> float:
    return i + 1


def zero_expr(a) -> float:
    return 0.0


class TestFeatureRatio(unittest.TestCase):
    def test_base_ratio(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        ratio_name = 'ratio'
        fd = ft.FeatureExpression('AddAmount', ft.FEATURE_TYPE_FLOAT, test_expr, [fa])
        fr = ft.FeatureRatio(ratio_name, ft.FEATURE_TYPE_FLOAT, fa, fd)
        with en.EnginePandas(num_threads=1) as e:
            td1 = ft.TensorDefinition('All', [fa, fd])
            df1 = e.df_from_csv(td1, file, inference=False)
            df1['ratio-2'] = df1[fa.name].div(df1[fd.name])
            td2 = ft.TensorDefinition('Derived', [fd, fr])
            df2 = e.df_from_csv(td2, file, inference=False)
            self.assertTrue(df2[fr.name].equals(df1['ratio-2']), f'Ratios not equal')
            self.assertEqual(df2.columns[1], ratio_name, f'Column name incorrect. Got {df2.columns[0]}')
            self.assertEqual(df2.iloc[:, 1].dtype.name, 'float64', f'Expecting a float data type')
            self.assertEqual(fr.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertEqual(fr.inference_ready, True, f'Expression feature should be ready for inference')
            self.assertEqual(len(fr.embedded_features), 2, f'Expecting 1 embedded ft. Got {len(fr.embedded_features)}')
            self.assertListEqual(
                sorted(fr.embedded_features, key=lambda x: x.name),
                sorted([fa, fd], key=lambda y: y.name), f'Embedded features should be fa, fr and fd'
            )
            self.assertEqual(td2.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertListEqual(
                sorted(td2.continuous_features(), key=lambda x: x.name),
                sorted([fd, fr], key=lambda y: y.name), f'Expanded Feature not correct')
            self.assertEqual(td2.rank, 2, f'This should have been a rank 2 tensor. Got {td2.rank}')
            self.assertEqual(len(td2.embedded_features), 3, f'Expecting 4 embedded feats {len(td2.embedded_features)}')
            self.assertListEqual(
                sorted(td2.embedded_features, key=lambda x: x.name),
                sorted([fa, fr, fd], key=lambda y: y.name), f'Embedded features should be fa, fr and fd'
            )

    def test_zero_denominator(self):
        # Test if the zero division return 0 instead of an error or np.inf
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        ratio_name = 'ratio'
        fd = ft.FeatureExpression('ZeroAmount', ft.FEATURE_TYPE_FLOAT, zero_expr, [fa])
        fr = ft.FeatureRatio(ratio_name, ft.FEATURE_TYPE_FLOAT, fa, fd)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fr])
            df = e.df_from_csv(td, file, inference=False)
            self.assertTrue((df[fr.name] == 0.0).all(), f'Ratios not all zero')
            self.assertEqual(df.columns[0], ratio_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'float64', f'Expecting a float data type')
            self.assertEqual(fr.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertEqual(fr.inference_ready, True, f'Expression feature should be ready for inference')
            self.assertEqual(len(fr.embedded_features), 2, f'Expecting 1 embedded ft. Got {len(fr.embedded_features)}')
            self.assertListEqual(
                sorted(fr.embedded_features, key=lambda x: x.name),
                sorted([fa, fd], key=lambda y: y.name), f'Embedded features should be fa, fr and fd'
            )
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertListEqual(
                sorted(td.continuous_features(), key=lambda x: x.name),
                sorted([fr], key=lambda y: y.name), f'Expanded Feature not correct')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(len(td.embedded_features), 3, f'Expecting 4 embedded feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fa, fr, fd], key=lambda y: y.name), f'Embedded features should be fa, fr and fd'
            )


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fd = ft.FeatureExpression('AddAmount', ft.FEATURE_TYPE_FLOAT, test_expr, [fa])
        fr = ft.FeatureRatio('Ratio', ft.FEATURE_TYPE_FLOAT, fa, fd)
        td = ft.TensorDefinition('TestNP', [fr])
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td, file, inference=False)
            n = e.np_from_csv(td, file, inference=False)
        self.assertEqual(type(n), en.TensorInstanceNumpy, f'Did not get TensorInstanceNumpy. But {type(n)}')
        self.assertEqual(len(n.numpy_lists), 1, f'Expected only one list. Got {len(n.numpy_lists)}')
        self.assertEqual(len(n), len(df), f'Lengths not equal {len(df)}, {len(n)}')
        self.assertTrue(np.all(np.equal(df.to_numpy(), n.numpy_lists[0])), f'from np not OK. {df}, {n.numpy_lists[0]}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
