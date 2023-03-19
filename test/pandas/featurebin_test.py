"""
Unit Tests for PandasNumpy Engine, specifically the FeatureBin usage
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


class TestFeatureBin(unittest.TestCase):
    def test_read_base_non_inference(self):
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        bin_name = 'Amount_bin'
        nr_bins = 3
        fb = ft.FeatureBin(bin_name, ft.FEATURE_TYPE_INT_16, fa, nr_bins)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fa])
            df = e.df_from_csv(td, file, inference=False)
            mx = np.finfo(df['Amount'].dtype).max
            mn = df['Amount'].min()
            md = df['Amount'].mean()
            td2 = ft.TensorDefinition('Derived', [fb])
            df = e.df_from_csv(td2, file, inference=False)
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(fb.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(df.columns[0], bin_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'category', f'Expecting a "category" data type')
            self.assertEqual(fb.inference_ready, True, f'Bin feature should be ready for inference')
            self.assertEqual(fb.bins, [mn, md, mx], f'Bins not set as expected. Got {fb.bins}')
            self.assertEqual(len(fb.embedded_features), 1, f'Expecting one embedded feature')
            self.assertListEqual(fb.embedded_features, [fa], f'Expecting fa to be the embedded feature')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(td2.inference_ready, True, f'TensorDefinition should be ready for inference')
            self.assertListEqual(td2.categorical_features(), [fb], f'Expanded Feature not correct')
            self.assertEqual(sorted(list(df[bin_name].unique())), list(range(0, fb.number_of_bins)))
            self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embedded feats {len(td2.embedded_features)}')
            self.assertListEqual(
                sorted(td2.embedded_features, key=lambda x: x.name),
                sorted([fb, fa], key=lambda x: x.name), f'Embedded features should be fb and fa'
            )

    def test_read_remove_element(self):
        # Test to see what happens if a small value is present during NON inference, but not during inference. Then
        # we should basically not be seeing that value in the output. In this case the bins during non inference will
        # be 0,1,1,2,2.
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        bin_name = 'Amount_bin'
        nr_bins = 3
        fb = ft.FeatureBin(bin_name, ft.FEATURE_TYPE_INT_16, fa, nr_bins)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the first line
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('Derived', [fa])
            df = e.df_from_csv(td, file1, inference=False)
            mx = np.finfo(df['Amount'].dtype).max
            mn = df['Amount'].min()
            md = df['Amount'].mean()
            td_d = ft.TensorDefinition('Derived', [fb])
            _ = e.df_from_csv(td_d, file1, inference=False)
            self.assertEqual(fb.inference_ready, True, f'Bin Feature should have been ready for inference')
            self.assertEqual(td_d.inference_ready, True, f'TensorDefinition should have been ready for inference')
            # Now run inference mode on file w/removed line
            df_1 = e.df_from_csv(td_d, file2, inference=True)
            bin_v = df_1[bin_name].unique()
            # Should not have a 0 bin. As we removed the 0.0 amount
            self.assertNotIn(0, bin_v, f'Should not have had a 0 value')
            self.assertEqual(len(df_1.columns), 1, f'Should have gotten one column. Got {len(df_1.columns)}')
            self.assertEqual(fb.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(df_1.columns[0], bin_name, f'Column name incorrect. Got {df_1.columns[0]}')
            self.assertEqual(df_1.iloc[:, 0].dtype.name, 'category', f'Expecting a "category" data type')
            self.assertEqual(fb.inference_ready, True, f'Bin feature should be ready for inference')
            self.assertEqual(fb.bins, [mn, md, mx], f'Bins probably changed. Got {fb.bins}')
            self.assertEqual(len(bin_v), fb.number_of_bins - 1, f'Not missing a value. Got len{len(bin_v)}')
            self.assertEqual(td_d.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertListEqual(td_d.categorical_features(), [fb], f'Expanded Feature not correct')
            self.assertEqual(len(td_d.embedded_features), 2, f'Expecting 2 embed feats {len(td_d.embedded_features)}')
            self.assertListEqual(
                sorted(td_d.embedded_features, key=lambda x: x.name),
                sorted([fb, fa], key=lambda x: x.name), f'Embedded features should be fb and fa'
            )

        remove_file_if_exists(file2)

    def test_read_add_element_small(self):
        # Test to validate what happens when during inference a value is seen that is smaller than any value seen during
        # non-inference. We would expect that to go in the smallest (0) bin
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        bin_name = 'Amount_bin'
        nr_bins = 3
        fb = ft.FeatureBin(bin_name, ft.FEATURE_TYPE_INT_16, fa, nr_bins)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the first line
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td_c = ft.TensorDefinition('All', [fa])
            df = e.df_from_csv(td_c, file2, inference=False)
            mx = np.finfo(df['Amount'].dtype).max
            mn = df['Amount'].min()
            md = df['Amount'].mean()
            # Run without inference on reduced file
            td = ft.TensorDefinition('Derived', [fb])
            df_n = e.df_from_csv(td, file2, inference=False)
            bin_v = df_n[bin_name].unique()
            # And inference on original file
            df = e.df_from_csv(td, file1, inference=True)
            self.assertEqual(fb.number_of_bins, len(bin_v), f'Number of bins changed {fb.number_of_bins}')
            self.assertEqual(df[bin_name][0], 0, f'Missing row should have been in 0 bin {df[bin_name][0]}')
            self.assertEqual(df[bin_name][1], 0, f'2 nd row should have been in 0 bin {df[bin_name][0]}')
            self.assertEqual(df[bin_name].iloc[-1], nr_bins-1, f'Last should have max bin {df[bin_name].iloc[-1]}')
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(fb.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(df.columns[0], bin_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'category', f'Expecting a "category" data type')
            self.assertEqual(fb.bins, [mn, md, mx], f'Bins not set as expected. Got {fb.bins}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertListEqual(td.categorical_features(), [fb], f'Expanded Feature not correct')
            self.assertEqual(sorted(list(df[bin_name].unique())), list(range(0, fb.number_of_bins)))
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embed feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fb, fa], key=lambda x: x.name), f'Embedded features should be fb and fa'
            )

        remove_file_if_exists(file2)

    def test_read_add_element_large(self):
        # Test to validate what happens when during inference a value is seen that is bigger than any value seen during
        # non-inference. We would expect that to go in the biggest bin
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        bin_name = 'Amount_bin'
        nr_bins = 3
        fb = ft.FeatureBin(bin_name, ft.FEATURE_TYPE_INT_16, fa, nr_bins)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the first line
        remove_file_if_exists(file2)
        copy_file_remove_last_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td_c = ft.TensorDefinition('All', [fa])
            df = e.df_from_csv(td_c, file2, inference=False)
            mx = np.finfo(df['Amount'].dtype).max
            mn = df['Amount'].min()
            md = df['Amount'].mean()
            # Run without inference on reduced file
            td = ft.TensorDefinition('Derived', [fb])
            df_n = e.df_from_csv(td, file2, inference=False)
            bin_v = df_n[bin_name].unique()
            # And inference on original file
            df = e.df_from_csv(td, file1, inference=True)
            self.assertEqual(fb.number_of_bins, len(bin_v), f'Number of bins changed {fb.number_of_bins}')
            self.assertEqual(df[bin_name][0], 0, f'First row should have been in 0 bin {df[bin_name][0]}')
            self.assertEqual(df[bin_name].iloc[-1], nr_bins-1, f'Last should have max bin {df[bin_name].iloc[-1]}')
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(fb.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Expecting Categorical LC')
            self.assertEqual(df.columns[0], bin_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'category', f'Expecting a "category" data type')
            self.assertEqual(fb.bins, [mn, md, mx], f'Bins not set as expected. Got {fb.bins}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td.categorical_features(), [fb], f'Expanded Feature not correct')
            self.assertEqual(sorted(list(df[bin_name].unique())), list(range(0, fb.number_of_bins)))
            self.assertEqual(len(td.embedded_features), 2, f'Expecting 2 embed feats {len(td.embedded_features)}')
            self.assertListEqual(
                sorted(td.embedded_features, key=lambda x: x.name),
                sorted([fb, fa], key=lambda x: x.name), f'Embedded features should be fb and fa'
            )

        remove_file_if_exists(file2)


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fb = ft.FeatureBin('BinAmount', ft.FEATURE_TYPE_INT_16, fa, 3)
        td = ft.TensorDefinition('TestNP', [fb])
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
