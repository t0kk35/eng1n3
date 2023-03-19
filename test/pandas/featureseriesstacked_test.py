"""
Unit Tests for PandasNumpy Engine, specifically the FeatureRatio usage
(c) 2023 d373c7
"""
import unittest
import f3atur3s as ft
import eng1n3.pandas as en

FILES_DIR = './files/'


class TestBaseCreate(unittest.TestCase):
    def test_create(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        fci = ft.FeatureIndex('Country_I', ft.FEATURE_TYPE_INT_16, fc)
        fmi = ft.FeatureIndex('MCC_I', ft.FEATURE_TYPE_INT_16, fm)
        fs = ft.FeatureSeriesStacked('Stacked_Categorical', ft.FEATURE_TYPE_INT_16, [fmi, fci],  3, fr)
#        fl = ft.FeatureLabelBinary('Fraud_Label', ft.FEATURE_TYPE_INT_8, ff)
        td2 = ft.TensorDefinition('Derived', [fs])
        with en.EnginePandas(num_threads=1) as e:
            s = e.np_from_csv(td2, file, time_feature=fd, inference=False)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
