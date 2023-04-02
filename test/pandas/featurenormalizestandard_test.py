"""
Unit Tests for PandasNumpy Engine, specifically the FeatureNormalizeStandard usage
(c) 2023 tsm
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


class TestNormalizeStandard(unittest.TestCase):
    """
    Derived Features. Normalize Standard feature tests. We'll use the from_df function for this
    """
    def test_read_base_non_inference(self):
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountStandard'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeStandard(s_name, s_type, fa)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fa])
            df1 = e.df_from_csv(td, file, inference=False)
            mn = df1['Amount'].mean()
            st = df1['Amount'].std()
            df1['standard'] = (df1['Amount'] - mn) / st
            td2 = ft.TensorDefinition('Derived', [fs])
            df2 = e.df_from_csv(td2, file, inference=False)
            self.assertEqual(len(df2.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df2.columns[0], s_name, f'Column name is not correct {df2.columns[0]}')
            self.assertTrue(df1['standard'].equals(df2[s_name]), 'Series not equal')
            self.assertEqual(df2.iloc[:, 0].dtype.name, 'float32', f'Expecting a "float32" data type')
            self.assertEqual(fs.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertListEqual(fs.embedded_features, [fa], 'Should have had fa as embedded features')
            self.assertEqual(fs.mean, mn, f'Mean not set correctly {fs.mean}')
            self.assertEqual(fs.stddev, st, f'Stddev not set correctly {fs.stddev}')
            self.assertEqual(td2.inference_ready, True, f'Tensor should be ready for inference')
            self.assertEqual(td2.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td2.continuous_features(), [fs], f'Expanded Feature not correct')
            self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embed feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td2.embedded_features, key=lambda x: x.name),
                sorted([fs, fa], key=lambda x: x.name), f'Embedded features should be fs and fa'
            )

    def test_float_down_cast(self):
        # Base feature is Float64 and will be down-cast to float32
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_64)
        s_name = 'AmountStandard'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeStandard(s_name, s_type, fa)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fa])
            df1 = e.df_from_csv(td, file, inference=False)
            mn = df1['Amount'].mean()
            st = df1['Amount'].std()
            df1['standard'] = (df1['Amount'] - mn) / st
            df1['standard'] = df1['standard'].astype('float32')
            td2 = ft.TensorDefinition('Derived', [fs])
            df2 = e.df_from_csv(td2, file, inference=False)
            self.assertEqual(len(df2.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df2.columns[0], s_name, f'Column name is not correct {df2.columns[0]}')
            self.assertTrue(df1['standard'].equals(df2[s_name]), 'Series not equal')
            self.assertEqual(df2.iloc[:, 0].dtype.name, 'float32', f'Expecting a "float32" data type')
            self.assertEqual(fs.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertListEqual(fs.embedded_features, [fa], 'Should have had fa as embedded features')
            self.assertEqual(fs.mean, mn, f'Mean not set correctly {fs.mean}')
            self.assertEqual(fs.stddev, st, f'Stddev not set correctly {fs.stddev}')
            self.assertEqual(td2.inference_ready, True, f'Tensor should be ready for inference')
            self.assertEqual(td2.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td2.continuous_features(), [fs], f'Expanded Feature not correct')
            self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embed feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td2.embedded_features, key=lambda x: x.name),
                sorted([fs, fa], key=lambda x: x.name), f'Embedded features should be fs and fa'
            )

    def test_read_w_logs_non_inference(self):
        # If logs are defined, then the amount should be logged first and then the normalizer applied.
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        s_name = 'AmountStandardAndLog'
        s_type = ft.FEATURE_TYPE_FLOAT
        file = FILES_DIR + 'engine_test_base_comma.csv'
        for log, log_fn in (('e', np.log), ('2', np.log2), ('10', np.log10)):
            with en.EnginePandas(num_threads=1) as e:
                fs = ft.FeatureNormalizeStandard(s_name, s_type, fa, log)
                td1 = ft.TensorDefinition('All', [fa])
                df1 = e.df_from_csv(td1, file, inference=False)
                mn = log_fn(df1['Amount']+fs.delta).mean()
                st = log_fn(df1['Amount']+fs.delta).std()
                df1['standard'] = (log_fn(df1['Amount']+fs.delta) - mn) / st
                td2 = ft.TensorDefinition('Derived', [fs])
                df2 = e.df_from_csv(td2, file, inference=False)
                self.assertEqual(len(df2.columns), 1, f'Only one columns should have been returned')
                self.assertEqual(df2.columns[0], s_name, f'Column name is not correct {df2.columns[0]}')
                self.assertTrue(df1['standard'].equals(df2[s_name]), f'Not equal {df1["standard"]}, {df2[s_name]}')
                self.assertEqual(df2.iloc[:, 0].dtype.name, 'float64', f'Expecting a "float32" data type')
                self.assertAlmostEqual(fs.mean, mn, 5, f'Means not set correctly {fs.mean}')
                self.assertAlmostEqual(fs.stddev, st, 5, f'Stddev not set correctly {fs.stddev}')
                self.assertListEqual(td2.continuous_features(True), [fs], f'Expanded Feature not correct')
                self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
                self.assertListEqual(fs.embedded_features, [fa], 'Should have had fa as embedded features')
                self.assertEqual(td2.inference_ready, True, f'Tensor should be ready for inference')
                self.assertEqual(td2.rank, 2, f'This should have been a rank 2 tensor. Got {td2.rank}')
                self.assertListEqual(td2.continuous_features(), [fs], f'Expanded Feature not correct')
                self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embed feats {len(td2.embedded_features)}')
                self.assertListEqual(
                    sorted(td2.embedded_features, key=lambda x: x.name),
                    sorted([fs, fa], key=lambda x: x.name), f'Embedded features should be fs and fa'
                )

    def test_inference_remove_element(self):
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountStandard'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeStandard(s_name, s_type, fa)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the first line
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fa])
            df_1 = e.df_from_csv(td, file1, inference=False)
            mn = df_1['Amount'].mean()
            st = df_1['Amount'].std()
            df_1['standard'] = (df_1['Amount'] - mn) / st
            df_1 = df_1.iloc[1:].reset_index()
            td = ft.TensorDefinition('Derived', [fs])
            _ = e.df_from_csv(td, file1, inference=False)
            self.assertEqual(fs.inference_ready, True, f'Scale feature should be ready for inference')
            self.assertEqual(td.inference_ready, True, f'The TensorDefinition should be ready for inference')
            # Now remove a line and run in inference mode
            df_2 = e.df_from_csv(td, file2, inference=True)
            self.assertEqual(len(df_2.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df_2.columns[0], s_name, f'Column name is not correct {df_2.columns[0]}')
            self.assertEqual(df_1['standard'].equals(df_2[s_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertEqual(df_2.iloc[:, 0].dtype.name, 'float32', f'Expecting a "float32" data type')
            self.assertEqual(fs.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertListEqual(fs.embedded_features, [fa], 'Should have had fa as embedded features')
            self.assertEqual(fs.mean, mn, f'Mean not set correctly {fs.mean}')
            self.assertEqual(fs.stddev, st, f'Stddev not set correctly {fs.stddev}')
            self.assertEqual(td.inference_ready, True, f'Tensor should be ready for inference')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td.continuous_features(), [fs], f'Expanded Feature not correct')
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embed feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fs, fa], key=lambda x: x.name), f'Embedded features should be fs and fa'
            )

        remove_file_if_exists(file2)

    def test_inference_add_element(self):
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountStandard'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeStandard(s_name, s_type, fa)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the first line
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fa])
            df_1 = e.df_from_csv(td, file2, inference=False)
            mn = df_1['Amount'].mean()
            st = df_1['Amount'].std()
            # We have the max and min from file 2 now apply to file1
            td = ft.TensorDefinition('Derived', [fa])
            df_1 = e.df_from_csv(td, file1, inference=False)
            df_1['standard'] = (df_1['Amount'] - mn) / st
            # Read file 2, in non-inference, then file with inference
            td = ft.TensorDefinition('Derived', [fs])
            _ = e.df_from_csv(td, file2, inference=False)
            self.assertEqual(fs.inference_ready, True, f'Scale feature should be ready for inference')
            self.assertEqual(td.inference_ready, True, f'The TensorDefinition should be ready for inference')
            df_2 = e.df_from_csv(td, file1, inference=True)
            self.assertEqual(len(df_2.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df_2.columns[0], s_name, f'Column name is not correct {df_2.columns[0]}')
            self.assertEqual(df_1['standard'].equals(df_2[s_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertEqual(df_2.iloc[:, 0].dtype.name, 'float32', f'Expecting a "float32" data type')
            self.assertEqual(fs.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Expecting Continuous LC')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertListEqual(fs.embedded_features, [fa], 'Should have had fa as embedded features')
            self.assertEqual(fs.mean, mn, f'Mean not set correctly {fs.mean}')
            self.assertEqual(fs.stddev, st, f'Minimum not set correctly {fs.stddev}')
            self.assertEqual(td.inference_ready, True, f'Tensor should be ready for inference')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td.continuous_features(), [fs], f'Expanded Feature not correct')
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embed feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fs, fa], key=lambda x: x.name), f'Embedded features should be fs and fa'
            )

        remove_file_if_exists(file2)


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fs = ft.FeatureNormalizeStandard('Standard', ft.FEATURE_TYPE_FLOAT, fa)
        td1 = ft.TensorDefinition('TestNP1', [fs])
        td2 = ft.TensorDefinition('TestNP1', [fs])
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td1, file, inference=False)
            n = e.np_from_csv(td2, file, inference=False)
        self.assertEqual(type(n), en.TensorInstanceNumpy, f'Did not get TensorInstanceNumpy. But {type(n)}')
        self.assertEqual(len(n.numpy_lists), 1, f'Expected only one list. Got {len(n.numpy_lists)}')
        self.assertEqual(len(n), len(df), f'Lengths not equal {len(df)}, {len(n)}')
        self.assertTrue(np.all(np.equal(df.to_numpy(), n.numpy_lists[0])), f'from np not OK. {df}, {n.numpy_lists[0]}')
        self.assertEqual(td1.rank, td2.rank, f'Expected Rank to be the same {td1.rank} {td2.rank}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
