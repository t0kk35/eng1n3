"""
(c) 2020 d373c7
"""
import unittest
import os

import numpy as np

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


class TestFeatureConcat(unittest.TestCase):
    def test_base_concat(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_STRING)
        concat_name = 'concat'
        fn = ft.FeatureConcat(concat_name, ft.FEATURE_TYPE_STRING, fc, fm)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fc, fm, fn])
            df = e.from_csv(td, file, inference=False)
            df['concat-1'] = df[fc.name].astype(str) + df[fm.name].astype(str)
            td_2 = ft.TensorDefinition('Derived', [fn])
            df_2 = e.from_csv(td_2, file, inference=False)
            self.assertTrue(df_2[fn.name].equals(df['concat-1']), f'Concat columns not equal')
            self.assertEqual(len(df_2.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(fn.learning_category, ft.LEARNING_CATEGORY_NONE, f'Expecting None LC')
            self.assertEqual(df_2.columns[0], concat_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df_2.iloc[:, 0].dtype.name, 'object', f'Expecting a "object" data type')
            self.assertEqual(fn.inference_ready, True, f'Concat feature should be ready for inference')
            self.assertEqual(len(fn.embedded_features), 2, f'Expecting Z embedded feature')
            self.assertListEqual(fn.embedded_features, [fc, fm], f'Expecting fa to be the embedded feature')
            self.assertEqual(td_2.rank, 2, f'This should have been a rank 2 tensor. Got {td_2.rank}')
            self.assertEqual(td_2.inference_ready, True, f'TensorDefinition should be ready for inference')
            self.assertEqual(len(td_2.embedded_features), 3, f'Expecting 3 embed feats {len(td_2.embedded_features)}')
            self.assertListEqual(
                sorted(td_2.embedded_features, key=lambda x: x.name),
                sorted([fn, fm, fc], key=lambda x: x.name), f'Embedded features should be fn, fm and fc'
            )

    def test_multiple_concat(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_STRING)
        concat_name_1 = 'concat_1'
        concat_name_2 = 'concat_2'
        fn1 = ft.FeatureConcat(concat_name_1, ft.FEATURE_TYPE_STRING, fc, fm)
        fn2 = ft.FeatureConcat(concat_name_2, ft.FEATURE_TYPE_STRING, fr, fm)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fc, fm, fr])
            df = e.from_csv(td, file, inference=False)
            df['concat-1'] = df[fc.name].astype(str) + df[fm.name].astype(str)
            df['concat-2'] = df[fr.name].astype(str) + df[fm.name].astype(str)
            td_1 = ft.TensorDefinition('Concat', [fn1, fn2])
            df_1 = e.from_csv(td_1, file, inference=False)
            self.assertTrue(df_1[fn1.name].equals(df['concat-1']), f'Concat 1 columns not equal')
            self.assertTrue(df_1[fn2.name].equals(df['concat-2']), f'Concat 2 columns not equal')
            self.assertEqual(len(df_1.columns), 2, f'Should have gotten one column. Got {len(df_1.columns)}')
            self.assertEqual(fn1.learning_category, ft.LEARNING_CATEGORY_NONE, f'Expecting None LC')
            self.assertEqual(fn2.learning_category, ft.LEARNING_CATEGORY_NONE, f'Expecting None LC')
            self.assertEqual(df_1.columns[0], concat_name_1, f'Column 3 name incorrect. Got {df_1.columns[0]}')
            self.assertEqual(df_1.columns[1], concat_name_2, f'Column 4 name incorrect. Got {df_1.columns[0]}')
            self.assertEqual(df_1.iloc[:, 0].dtype.name, 'object', f'Expecting a "object" data type for col1')
            self.assertEqual(df_1.iloc[:, 1].dtype.name, 'object', f'Expecting a "object" data type for col2')
            self.assertEqual(fn1.inference_ready, True, f'Concat feature should be ready for inference')
            self.assertEqual(fn2.inference_ready, True, f'Concat feature should be ready for inference')
            self.assertEqual(len(fn1.embedded_features), 2, f'Expecting Z embedded feature')
            self.assertListEqual(fn1.embedded_features, [fc, fm], f'Expecting fa to be the embedded feature')
            self.assertEqual(len(fn2.embedded_features), 2, f'Expecting Z embedded feature')
            self.assertListEqual(fn2.embedded_features, [fr, fm], f'Expecting fa to be the embedded feature')
            self.assertEqual(td_1.rank, 2, f'This should have been a rank 2 tensor. Got {td_1.rank}')
            self.assertEqual(td_1.inference_ready, True, f'TensorDefinition should be ready for inference')
            self.assertEqual(len(td_1.embedded_features), 5, f'Expecting 3 embed feats {len(td_1.embedded_features)}')
            self.assertListEqual(
                sorted(td_1.embedded_features, key=lambda x: x.name),
                sorted([fr, fm, fc, fn1, fn2], key=lambda x: x.name), f'Embedded features should be fn, fm and fc'
            )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
