"""
Unit Tests for Profile Package
(c) 2022 d373c7
"""
import unittest
import statistics as stat

import f3atur3s as ft

from eng1n3.profile import ProfileElementNative


FILES_DIR = './files/'


def card_is_1(x: str) -> bool:
    return x == 'CARD-1'


def card_is_2(x: str) -> bool:
    return x == 'CARD-2'


def add_one(x: float) -> float:
    return x + 1


def always_false(x: str) -> float:
    return False


class TestNativeElement(unittest.TestCase):
    # Define some dummy FeatureGroupers
    tp = ft.TIME_PERIOD_DAY
    fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
    fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
    fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
    fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
    fgs = ft.FeatureGrouper('S', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_SUM)
    fga = ft.FeatureGrouper('A', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_AVG)
    fgt = ft.FeatureGrouper('ST', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_STDDEV)
    fgx = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_MAX)
    fgm = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_MIN)

    def test_base_native_element(self):
        lst = [1.0, 2.0, 2.5, 3, 5, 5.5]
        pe = ProfileElementNative()
        for e in lst:
            pe.contribute(e)
        x = len(lst)
        y = pe.aggregate(self.fgc)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst)
        y = pe.aggregate(self.fgs)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst)
        y = pe.aggregate(self.fga)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst)
        y = pe.aggregate(self.fgt)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst)
        y = pe.aggregate(self.fgx)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst)
        y = pe.aggregate(self.fgm)
        self.assertEqual(x, y, f'Minimums should have been equal {x} {y}')

    def test_element_merge_native_element(self):
        lst1 = [1.0, 2.0, 2.5, 3, 5, 5.5]
        lst2 = [6.0, 7.5, 8.0]
        lst3 = lst1 + lst2
        pe1 = ProfileElementNative()
        for e in lst1:
            pe1.contribute(e)
        pe2 = ProfileElementNative()
        for e in lst2:
            pe2.contribute(e)
        pe3 = ProfileElementNative()
        pe3.merge(pe1)
        pe3.merge(pe2)
        x = len(lst3)
        y = pe3.aggregate(self.fgc)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst3)
        y = pe3.aggregate(self.fgs)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst3)
        y = pe3.aggregate(self.fga)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst3)
        y = pe3.aggregate(self.fgt)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst3)
        y = pe3.aggregate(self.fgx)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst3)
        y = pe3.aggregate(self.fgm)
        self.assertEqual(x, y, f'Minimums should have been equal {x} {y}')

    def merge_empty_element(self):
        lst1 = [1.0, 2.0, 2.5, 3, 5, 5.5]
        pe1 = ProfileElementNative()
        for e in lst1:
            pe1.contribute(e)
        pe2 = ProfileElementNative()
        pe3 = ProfileElementNative()
        pe3.merge(pe1)
        pe3.merge(pe2)
        x = len(lst1)
        y = pe3.aggregate(self.fgc)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst1)
        y = pe3.aggregate(self.fgs)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst1)
        y = pe3.aggregate(self.fga)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst1)
        y = pe3.aggregate(self.fgt)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst1)
        y = pe3.aggregate(self.fgx)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst1)
        y = pe3.aggregate(self.fgm)
        self.assertEqual(x, y, f'Minimums should have been equal {x} {y}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
