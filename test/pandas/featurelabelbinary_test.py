"""
Unit Tests for PandasNumpy Engine, specifically the FeatureIndex usage
(c) 2020 d373c7
"""
import unittest

import numpy as np

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


class TestIndex(unittest.TestCase):
    """Index Feature Testcases"""
    def test_read_base(self):
        fa = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_INT_16)
        label_name = 'Fraud_Label'
        fl = ft.FeatureLabelBinary(label_name, ft.FEATURE_TYPE_INT_8, fa)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fl])
            df = e.df_from_csv(td, file, inference=False)
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(fl.learning_category, ft.LEARNING_CATEGORY_LABEL, f'Expecting Label LC')
            self.assertEqual(df.columns[0], label_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'int8', f'Expecting abn "int8" data type')
            self.assertEqual(fl.inference_ready, True, f'Bin feature should be ready for inference')
            self.assertEqual(len(fl.embedded_features), 1, f'Expecting one embedded feature')
            self.assertListEqual(fl.embedded_features, [fa], f'Expecting fa to be the embedded feature')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(td.inference_ready, True, f'TensorDefinition should be ready for inference')
            self.assertListEqual(td.label_features(), [fl], f'Expanded Feature not correct')
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embedded feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fl, fa], key=lambda x: x.name), f'Embedded features should be fl and fa'
            )

    # TODO Should add a test for value other than 0 and 1.


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_INT_16)
        fl = ft.FeatureLabelBinary('FraudLabel', ft.FEATURE_TYPE_INT_8, fa)
        td1 = ft.TensorDefinition('TestNP1', [fl])
        td2 = ft.TensorDefinition('TestNP2', [fl])
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
