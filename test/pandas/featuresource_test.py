"""
Unit Tests for PandasNumpy Engine, specifically the FeatureSource usage
(c) 2020 d373c7
"""
import unittest
import f3atur3s as ft

import eng1n3.pandas.pandasengine as en

FILES_DIR = './files/'


class TestReading(unittest.TestCase):
    """Base Reading Tests
    """
    fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL)
    features = [
        (ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d'), 'DateTime'),
        (ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT), 'Float'),
        (ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING), 'String'),
        (fc, 'Categorical')
    ]

    def test_creation_base(self):
        threads = 1
        with en.EnginePandas(num_threads=1) as e:
            self.assertIsInstance(e, en.EnginePandas, 'PandasNumpy engine creation failed')
            self.assertEqual(e.num_threads, threads, f'Num_threads incorrect got {e.num_threads}')

    def test_read_base_single(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            for f, d in TestReading.features:
                td = ft.TensorDefinition(d, [f])
                df = e.from_csv(td, file, inference=False)
                self.assertEqual(len(df.columns), 1, f'Expected a one column panda for read test {d}')
                self.assertEqual(df.columns[0], f.name, f'Wrong panda column for read test {d}. Got {df.columns[0]}')
                self.assertEqual(td.inference_ready, True, 'TensorDefinition Should have been ready for inference')

    def test_read_base_all(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            td = ft.TensorDefinition('All', [f for f, d in TestReading.features])
            df = e.from_csv(td, file, inference=False)
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for (f, _), c in zip(TestReading.features, df.columns):
                self.assertEqual(f.name, c, f'Incorrect column name in read test all got {c}, expected {f.name}')

    def test_read_base_all_non_def_delimiter(self):
        file = FILES_DIR + 'engine_test_base_pipe.csv'
        with en.EnginePandas(num_threads=1) as e:
            df = e.from_csv(ft.TensorDefinition('All', [f for f, d in TestReading.features]),
                            file, inference=False, delimiter='|')
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for (f, _), c in zip(TestReading.features, df.columns):
                self.assertEqual(f.name, c, f'Incorrect column name in read test all got {c}, expected {f.name}')

    # TODO need test with multiple source features that are dates. There was an iterator problem?


class TestDate(unittest.TestCase):
    """
    Some date specific tests
    """
    pass


class TestCategorical(unittest.TestCase):
    """
    Tests for categorical source feature
    """
    def test_is_categorical(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            self.assertEqual(df.dtypes[0], 'category', 'Source with f_type should be categorical')

    def test_categorical_default(self):
        default = 'DEF'
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default=default)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandas(num_threads=1) as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            self.assertEqual(fc.default, default, f'Default incorrect. Got {fc.default}')
            self.assertIn(default, list(df['MCC']), f'Default not found in Panda')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
