"""
Unit Tests for PandasNumpy Engine, specifically the FeatureOneIndex usage
(c) 2020 d373c7
"""
import unittest
import os

import numpy as np

import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


def remove_file_if_exists(file):
    try:
        os.remove(file)
    except OSError:
        pass


def copy_file_remove_last_line(file, new_file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
    with open(new_file, 'w') as fp:
        # iterate each line
        for number, line in enumerate(lines):
            if number < len(lines) - 1:
                fp.write(line)


def copy_file_remove_first_line(file, new_file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
    with open(new_file, 'w') as fp:
        # iterate each line
        for number, line in enumerate(lines):
            # Numbers are 0 based and there is a header, so line number 1 is the first line with data.
            if number != 1:
                fp.write(line)


class TestIndex(unittest.TestCase):
    """Index Feature Testcases"""
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        ind_name = 'MCC_ID'
        fi = ft.FeatureIndex(ind_name, ft.FEATURE_TYPE_INT_16, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td, file, inference=False)
            mcc_v = df['MCC'].unique()
            td2 = ft.TensorDefinition('Derived', [fi])
            df = e.from_csv(td2, file, inference=False)
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(df.columns[0], ind_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'int16', f'Expecting a uint16 data type')
            self.assertEqual(fi.inference_ready, True, f'Index feature should be ready for inference')
            self.assertEqual(fi.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(len(fi.embedded_features), 1, f'Expecting 1 embedded ft. Got {len(fi.embedded_features)}')
            self.assertEqual(fi.embedded_features[0], fc, 'Embedded feature should be the original source feature')
            self.assertEqual(len(fi.dictionary), len(mcc_v), f'Dictionary length not correct {len(fi.dictionary)}')
            self.assertEqual(len(fi), len(mcc_v), f'Dictionary length not correct {len(fi.dictionary)}')
            self.assertListEqual(list(fi.dictionary.keys()), list(mcc_v), f'Dictionary values don not match')
            self.assertEqual(td.inference_ready, True, f'Tensor should have been ready for inference now')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td2.categorical_features(True), [fi], f'Expanded Feature not correct')
            self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embedded feats {len(td2.embedded_features)}')
            self.assertListEqual(
                sorted(td2.embedded_features, key=lambda x: x.name),
                sorted([fi, fc], key=lambda x: x.name), f'Embedded features should be fi and fc'
            )

    def test_inference_read_remove_element(self):
        # If a value of an index feature is present in NON Inference mode, and it is not present during inference,
        # then that value should not be used, but present in the dictionary. If the first line of the
        # inference file is deleted, the value '1' should be missing.
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        ind_name = 'MCC_ID'
        fi = ft.FeatureIndex(ind_name, ft.FEATURE_TYPE_INT_16, fc)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the first line
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('Derived', [fi])
            df_1 = e.from_csv(td, file1, inference=False)
            self.assertEqual(fi.inference_ready, True, f'Index feature should be ready for inference')
            self.assertEqual(td.inference_ready, True, f'The TensorDefinition should be ready for inference')
            # Now remove a line and run in inference mode
            df_1 = df_1.iloc[1:].reset_index()
            df_2 = e.from_csv(td, file2, inference=True)
            mcc_v = df_2[ind_name].unique()
            self.assertTrue(1 in fi.dictionary.values(), f'The value 1 should be in the dictionary')
            self.assertFalse(1 in mcc_v, f'The Value 1 should not appear in the inference set')
            self.assertEqual(df_1[ind_name].equals(df_2[ind_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(len(fi), len(mcc_v) + 1, f'Length of dictionary changed {len(fi)}')
            self.assertEqual(fi.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(len(fi.embedded_features), 1, f'Expecting 1 embedded ft. Got {len(fi.embedded_features)}')
            self.assertEqual(fi.embedded_features[0], fc, 'Embedded feature should be the original source feature')
            self.assertEqual(len(fi.dictionary), len(mcc_v) + 1, f'Dictionary length not correct {len(fi.dictionary)}')
            self.assertEqual(len(fi), len(mcc_v) + 1, f'Dictionary length not correct {len(fi.dictionary)}')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertListEqual(td.categorical_features(True), [fi], f'Expanded Feature not correct')
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embedded feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fi, fc], key=lambda x: x.name), f'Embedded features should be fi and fc'
            )

        remove_file_if_exists(file2)

    def test_inference_read_add_element(self):
        # If and index value is present in the file when running in inference, but it was not in NON inference, then
        # the default 0 value should be assigned.
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        ind_name = 'MCC_ID'
        fi = ft.FeatureIndex(ind_name, ft.FEATURE_TYPE_INT_16, fc)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the first line
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('Derived', [fi])
            df_1 = e.from_csv(td, file2, inference=False)
            mcc_v = df_1[ind_name].unique()
            # And inference on original
            df_2 = e.from_csv(td, file1, inference=True)
            self.assertTrue(1 in fi.dictionary.values(), f'The value 1 should not be in the dictionary')
            self.assertTrue(1 in mcc_v, f'The Value 1 should not appear in the inference set')
            self.assertTrue(df_1[ind_name].equals(df_2.iloc[1:].reset_index()[ind_name]), f'Series not the same')
            self.assertEqual(df_2[ind_name][0], 0, f'Missing row should have been default {df_2[ind_name][0]}')
            self.assertEqual(fi.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(len(fi.embedded_features), 1, f'Expecting 1 embedded ft. Got {len(fi.embedded_features)}')
            self.assertEqual(fi.embedded_features[0], fc, 'Embedded feature should be the original source feature')
            self.assertEqual(len(fi.dictionary), len(mcc_v), f'Length of dictionary changed {len(fi)}')
            self.assertEqual(len(fi), len(mcc_v), f'Length of dictionary changed {len(fi)}')
            self.assertTrue(np.all(np.isin(mcc_v, list(fi.dictionary.values()))))
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertListEqual(td.categorical_features(True), [fi], f'Expanded Feature not correct')
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embedded feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fi, fc], key=lambda x: x.name), f'Embedded features should be fi and fc'
            )

        remove_file_if_exists(file2)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
