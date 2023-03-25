"""
Unit Tests for PandasNumpy Engine, specifically the FeatureOneHot usage
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


class TestOneHot(unittest.TestCase):
    """ Derived Features. One Hot feature tests """
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [fc])
            df1 = e.df_from_csv(td, file, inference=False)
            mcc_v = df1['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            td2 = ft.TensorDefinition('Derived', [fo])
            df = e.df_from_csv(td2, file, inference=False)
            # x = df.iloc[:, 0].dtype.name
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(df.iloc[:, 0].dtype.name, 'uint8', f'Expecting a uint8 data type')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertEqual(len(fo.embedded_features), 1, f'Expecting 1 embedded ft. Got {len(fo.embedded_features)}')
            self.assertEqual(fo.embedded_features[0], fc, 'Embedded feature should be the original source feature')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')
            vt = set([vf for vf in fo.expand()])
            self.assertEqual(len(vt), len(mcc_v), f'Should have gotten {len(mcc_v)} expanded features')
            self.assertIsInstance(vt.pop(), ft.FeatureVirtual, f'Expanded features should be Virtual Features')
            vn = [vf.name for vf in fo.expand()]
            self.assertListEqual(vn, mcc_c, f'Names of the Virtual Features must match columns')
            self.assertEqual(td.inference_ready, True, f'Tensor should have been ready for inference now')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertEqual(fo.learning_category, ft.LEARNING_CATEGORY_BINARY, f'Expecting Binary Learning type')
            self.assertListEqual(td2.binary_features(True), fo.expand(), f'Expanded Feature not correct')
            self.assertEqual(len(td2.embedded_features), 2, f'Expecting 2 embedded feats {len(td2.embedded_features)}')
            self.assertListEqual(
                sorted(td2.embedded_features, key=lambda x: x.name),
                sorted([fc, fo], key=lambda x: x.name), f'Embedded features should be fo and fc'
            )

    # TODO Run test to check that a run with inference can only be done with a td that is ready for inference

    def test_read_base_inference_removed_element(self):
        # If a value is a file read without inference and that value is NOT in a file read WITH inference, then
        # there should still be an expanded column with the name of the missing value.
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the last line
        remove_file_if_exists(file2)
        copy_file_remove_last_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td_c = ft.TensorDefinition('All', [fc])
            df_c = e.df_from_csv(td_c, file1, inference=False)
            mcc_v = df_c['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            td_o = ft.TensorDefinition('OH', [fo])
            _ = e.df_from_csv(td_o, file1, inference=False)
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should be read for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            # Now Read file2 in inference mode. It should have all one_hot values from file1
            df = e.df_from_csv(td_o, file2, inference=True)
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should still be ready for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be still ready for inference')
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')
            self.assertEqual(td_o.rank, 2, f'This should have been a rank 2 tensor. Got {td_o.rank}')
            self.assertListEqual(td_o.binary_features(True), fo.expand(), f'Expanded Feature not correct')
            self.assertEqual(len(td_o.embedded_features), 2, f'Expecting 2 embedded feat {len(td_o.embedded_features)}')
            self.assertListEqual(
                sorted(td_o.embedded_features, key=lambda x: x.name),
                sorted([fc, fo], key=lambda x: x.name), f'Embedded features should be fo and fc'
            )

        remove_file_if_exists(file2)

    def test_read_base_inference_added_element(self):
        # If a value of a OneHot Feature is NOT in the file without inference, but that value exists during inference,
        # then that value should be removed. The number of expanded features should be the same regardless of inference
        # We are going to remove the first line as the last happens to be the default which is always added.
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td_c = ft.TensorDefinition('All', [fc])
            df_c = e.df_from_csv(td_c, file2, inference=False)
            mcc_v = df_c['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            td_o = ft.TensorDefinition('OH', [fo])
            _ = e.df_from_csv(td_o, file2, inference=False)
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should be read for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            # Now Read file1 in inference mode. It should have less one hot values than are in file1
            df = e.df_from_csv(td_o, file1, inference=True)
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should still be ready for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be still ready for inference')
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')
            self.assertEqual(td_o.rank, 2, f'This should have been a rank 2 tensor. Got {td_o.rank}')
            self.assertListEqual(td_o.binary_features(True), fo.expand(), f'Expanded Feature not correct')
            self.assertEqual(len(td_o.embedded_features), 2, f'Expecting 2 embedded feat {len(td_o.embedded_features)}')
            self.assertListEqual(
                sorted(td_o.embedded_features, key=lambda x: x.name),
                sorted([fc, fo], key=lambda x: x.name), f'Embedded features should be fo and fc'
            )

        remove_file_if_exists(file2)

    def test_read_base_inference_removed_default(self):
        # If a value of a OneHot Feature is NOT in the file without inference, but that value exists during inference,
        # then that value should be removed. That is, except for the default. The default should always be
        # present, even with all 0
        _ = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the last line. The last line happens to be the line
        # with the default.
        remove_file_if_exists(file2)
        copy_file_remove_last_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td_c = ft.TensorDefinition('All', [fc])
            df_c = e.df_from_csv(td_c, file2, inference=False)
            mcc_v = df_c['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            # Add the default
            mcc_c.append('MCC' + '__' + fc.default)
            td_o = ft.TensorDefinition('OH', [fo])
            _ = e.df_from_csv(td_o, file2, inference=False)
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should be read for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            # Now Read file2 in inference mode. It should have all one_hot values from file1
            df = e.df_from_csv(td_o, file1, inference=True)
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should still be ready for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be still ready for inference')
            self.assertEqual(len(df.columns), len(mcc_c), f'Col number must match values {len(df.columns), len(mcc_c)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')
            self.assertEqual(td_o.rank, 2, f'This should have been a rank 2 tensor. Got {td_o.rank}')
            self.assertListEqual(td_o.binary_features(True), fo.expand(), f'Expanded Feature not correct')
            self.assertEqual(len(td_o.embedded_features), 2, f'Expecting 2 embedded feat {len(td_o.embedded_features)}')
            self.assertListEqual(
                sorted(td_o.embedded_features, key=lambda x: x.name),
                sorted([fc, fo], key=lambda x: x.name), f'Embedded features should be fo and fc'
            )

        remove_file_if_exists(file2)

    def test_with_other_columns(self):
        # Regression Test, there was a bug that when the one-hot processor wanted to add a column seen in non-inference
        # mode, and that column was not present, then it tried to convert all items in the df to int8.
        # Which obviously fails on date and strings and such.
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file1 = FILES_DIR + 'engine_test_base_comma.csv'
        file2 = FILES_DIR + 'engine_test_base_comma_remove_line.csv'
        # Create duplicate of the base_file, but remove the last line. The last line happens to be the line
        # with the default.
        remove_file_if_exists(file2)
        copy_file_remove_first_line(file1, file2)
        with en.EnginePandas(num_threads=1) as e:
            td_c = ft.TensorDefinition('All', [fc])
            df_c = e.df_from_csv(td_c, file1, inference=False)
            mcc_v = df_c['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            td_o = ft.TensorDefinition('OH', [fd, fo])
            _ = e.df_from_csv(td_o, file1, inference=False)
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should be read for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c)
            # Now Read file2 in inference mode. It should have all one_hot values from file1
            df = e.df_from_csv(td_o, file2, inference=True)
            # Drop the Date Column
            df = df.drop(['Date'], axis=1)
            # Run standard tests.
            self.assertEqual(td_o.inference_ready, True, f'The TensorDefinition should still be ready for inference')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be still ready for inference')
            self.assertEqual(len(df.columns), len(mcc_c), f'Col number must match values {len(df.columns), len(mcc_c)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')
            self.assertEqual(td_o.rank, 2, f'This should have been a rank 2 tensor. Got {td_o.rank}')
            self.assertListEqual(td_o.binary_features(True), fo.expand(), f'Expanded Feature not correct')
            self.assertEqual(len(td_o.embedded_features), 3, f'Expecting 2 embedded feat {len(td_o.embedded_features)}')
            self.assertListEqual(
                sorted(td_o.embedded_features, key=lambda x: x.name),
                sorted([fd, fc, fo], key=lambda x: x.name), f'Embedded features should be fo and fc'
            )

        remove_file_if_exists(file2)


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL)
        fo = ft.FeatureOneHot('Country_OH', ft.FEATURE_TYPE_INT_16, fa)
        td1 = ft.TensorDefinition('TestNP1', [fo])
        td2 = ft.TensorDefinition('TestNP2', [fo])
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td1, file, inference=False)
            n = e.np_from_csv(td2, file, inference=False)
        self.assertEqual(type(n), en.TensorInstanceNumpy, f'Did not get TensorInstanceNumpy. But {type(n)}')
        self.assertEqual(len(n.numpy_lists), 1, f'Expected only one list. Got {len(n.numpy_lists)}')
        self.assertEqual(len(n), len(df), f'Lengths not equal {len(df)}, {len(n)}')
        self.assertTrue(np.all(np.equal(df.to_numpy(), n.numpy_lists[0])), f'from np not OK. {df}, {n.numpy_lists[0]}')
        self.assertEqual(td1.rank, td2.rank, f'Expected rank to be the same {td1.rank} {td2.rank}')

# TODO test default should always be included even if not in the file, not sure that is the case at present.


def main():
    unittest.main()


if __name__ == '__main__':
    main()
