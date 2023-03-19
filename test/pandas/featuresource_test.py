"""
Unit Tests for PandasNumpy Engine, specifically the FeatureSource usage
(c) 2020 tsm
"""
import unittest

import numpy as np

import f3atur3s as ft

import eng1n3.pandas.pandasengine as en

FILES_DIR = './files/'


class TestReading(unittest.TestCase):
    """Base Reading Tests
    """
    features = [
        (ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d'), 'DateTime', np.dtype('datetime64[ns]')),
        (ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT), 'Float', np.dtype('float64')),
        (ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING), 'String', np.dtype('object')),
    ]

    def test_creation_base(self):
        threads = 1
        with en.EnginePandas(num_threads=1) as e:
            self.assertIsInstance(e, en.EnginePandas, 'PandasNumpy engine creation failed')
            self.assertEqual(e.num_threads, threads, f'Num_threads incorrect got {e.num_threads}')

    def test_read_base_single(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            for f, d, t in TestReading.features:
                td = ft.TensorDefinition(d, [f])
                df = e.df_from_csv(td, file, inference=False)
                self.assertEqual(len(df.columns), 1, f'Expected a one column panda for read test {d}')
                self.assertEqual(df.columns[0], f.name, f'Wrong panda column for read test {d}. Got {df.columns[0]}')
                self.assertEqual(df[f.name].dtype, t, f'Unexpected type. Got <{df[f.name].dtype}>. Expected <{t}> ')
                self.assertEqual(td.inference_ready, True, 'TensorDefinition Should have been ready for inference')

    def test_read_base_all(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [f for f, d, t in TestReading.features])
            df = e.df_from_csv(td, file, inference=False)
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for i, (f, _, t) in enumerate(TestReading.features):
                self.assertEqual(f.name, df.columns[i], f'Incorrect column name got {df.columns[i]}, expected {f.name}')
                self.assertEqual(t, df[f.name].dtype, f'Unexpected type. Got <{df[f.name].dtype}>. Expected <{t}> ')

    def test_read_base_all_non_def_delimiter(self):
        file = FILES_DIR + 'engine_test_base_pipe.csv'
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(ft.TensorDefinition('All', [f for f, d, t in TestReading.features]),
                               file, inference=False, delimiter='|')
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for (f, _, t), c in zip(TestReading.features, df.columns):
                self.assertEqual(f.name, c, f'Incorrect column name in read test all got {c}, expected {f.name}')

    def test_read_base_non_existent_column_bad(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        f = ft.FeatureSource('IDontExist', ft.FEATURE_TYPE_STRING)
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [f])
            with self.assertRaises(ValueError):
                _ = e.df_from_csv(td, file, inference=False)


class TestDate(unittest.TestCase):
    """
    Some date specific tests
    """
    def test_bad_format(self):
        f = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='bad')
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [f])
            with self.assertRaises(ValueError):
                _ = e.df_from_csv(td, file, inference=False)

    # TODO need test with multiple source features that are dates. There was an iterator problem?


class TestCategorical(unittest.TestCase):
    """
    Tests for categorical source feature
    """
    def test_is_categorical(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            self.assertEqual(df.dtypes[0], 'category', 'Source with f_type should be categorical')
            self.assertListEqual(df[fc.name].unique().dropna().tolist(), list(df.dtypes[0].categories))

    def test_categorical_default(self):
        default = 'DEF'
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default=default)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            self.assertEqual(fc.default, default, f'Default incorrect. Got {fc.default}')
            self.assertEqual(df.dtypes[0], 'category', 'Source with f_type should be categorical')
            self.assertIn(default, list(df['MCC']), f'Default not found in Panda')
            self.assertIn(default, list(df.dtypes[0].categories), f'Pandas column categories should contain default ')


class TestNP(unittest.TestCase):
    def test_from_np_good(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fs = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        td = ft.TensorDefinition('TestNP', [fs])
        with en.EnginePandas(num_threads=1) as e:
            df = e.df_from_csv(td, file, inference=False)
            n = e.np_from_csv(td, file, inference=False)
        self.assertEqual(type(n), en.TensorInstanceNumpy, f'Did not get TensorInstanceNumpy. But {type(n)}')
        self.assertEqual(len(n.numpy_lists), 1, f'Expected only one list. Got {len(n.numpy_lists)}')
        self.assertEqual(len(n), len(df), f'Lengths not equal {len(df)}, {len(n)}')
        self.assertTrue(np.all(np.equal(df.to_numpy(), n.numpy_lists[0])), f'from np not OK. {df}, {n.numpy_lists[0]}')

    def test_from_np_bad(self):
        # Should fail for LEARNING_CATEGORY_NONE
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fs = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        td = ft.TensorDefinition('TestNP', [fs])
        with en.EnginePandas(num_threads=1) as e:
            with self.assertRaises(en.EnginePandasException):
                _ = e.np_from_csv(td, file, inference=False)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
