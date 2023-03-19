"""
Unit Tests for TensorInstance class. The data that will go into the models
(c) 2023 tsm
"""
import unittest
import numpy as np

import f3atur3s as ft
import eng1n3.pandas as en


class TestTensorInstanceBase(unittest.TestCase):
    def test_creation_base(self):
        x = np.arange(10)
        y = np.arange(10)
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        self.assertIsInstance(n, en.TensorInstanceNumpy)
        self.assertEqual(len(n), len(x), f'Length not correct {len(n)}/{len(x)}')
        self.assertEqual(len(n.shapes[0]), 1, f'Shape should only have 1 dim {len(n.shapes[0])}')
        self.assertEqual(n.shapes[0][0], len(x), f'Shape of dim 0 incorrect {n.shapes[0][0]}')
        self.assertEqual(len(n.shapes[1]), 1, f'Shape should only have 1 dim {len(n.shapes[1])}')
        self.assertEqual(n.shapes[1][0], len(y), f'Shape of dim 0 incorrect {n.shapes[1][0]}')
        self.assertEqual(n.number_of_lists, len(c), f'Number of lists incorrect {n.number_of_lists}')
        self.assertEqual(n.dtype_names[0], x.dtype.name, f'dtype not expected {n.dtype_names[0]}')
        self.assertEqual(n.dtype_names[1], y.dtype.name, f'dtype not expected {n.dtype_names[1]}')
        self.assertTupleEqual(n.numpy_lists, c, f'Not the expected return from numpy_list {n.numpy_lists}')

    def test_creation_wrong_size(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.random.rand(5, 2)
        # This is a smaller numpy. The batch size (first dimension) is smaller.
        y = np.random.rand(2, 2)
        with self.assertRaises(en.TensorInstanceException):
            en.TensorInstanceNumpy((td,), (x, y))

    def test_lists(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        self.assertEqual(len(n.numpy_lists), len(c), f'Number of lists does not add up {len(n.numpy_lists)}')
        self.assertEqual((n.numpy_lists[0] == x).all(), True, f'Lists not equal')
        self.assertEqual((n.numpy_lists[1] == y).all(), True, f'Lists not equal')

    def test_slice_good(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        x0, y0 = n[0].numpy_lists
        self.assertEqual(np.array(x0 == x[0]).all(), True, f'First entries do not match {x0}, {x[0]}')
        self.assertEqual(np.array(y0 == y[0]).all(), True, f'First entries do not match {y0}, {y[0]}')
        x1, y1 = n[1].numpy_lists
        self.assertEqual(np.array(x1 == x[1]).all(), True, f'Second entries do not match {x1}, {x[1]}')
        self.assertEqual(np.array(y1 == y[1]).all(), True, f'Second entries do not match {y1}, {y[1]}')
        xf, yf = n[0:5].numpy_lists
        self.assertEqual(np.array(xf == x).all(), True, f'All entries do not match {xf}, {x}')
        self.assertEqual(np.array(yf == y).all(), True, f'All entries do not match {yf}, {y}')
        xm, ym = n[1:4].numpy_lists
        self.assertEqual(np.array(xm == x[1:4]).all(), True, f'Mid entries do not match {xf}, {x[1:4]}')
        self.assertEqual(np.array(ym == y[1:4]).all(), True, f'Mid entries do not match {yf}, {y[1:4]}')
        xl, yl = n[4].numpy_lists
        self.assertEqual(np.array(xl == x[-1]).all(), True, f'Last entries do not match {xl}, {x[-1]}')
        self.assertEqual(np.array(yl == y[-1]).all(), True, f'Last entries do not match {yl}, {y[-1]}')
        xl, yl = n[-1].numpy_lists
        self.assertEqual(np.array(xl == x[-1]).all(), True, f'Last entries do not match {xl}, {x[-1]}')
        self.assertEqual(np.array(yl == y[-1]).all(), True, f'Last entries do not match {yl}, {y[-1]}')

    def test_slice_bad(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        with self.assertRaises(en.TensorInstanceException):
            _ = n[5]
        with self.assertRaises(en.TensorInstanceException):
            _ = n[-6]
        with self.assertRaises(en.TensorInstanceException):
            _ = n[0:6]
        with self.assertRaises(en.TensorInstanceException):
            _ = n[-1:5]

    def test_unique(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.random.rand(5, 2)
        y = np.random.randint(32, size=(5, 2))
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        vl_1, cn_1 = np.unique(y, return_counts=True)
        vl_1, cn_1 = tuple(vl_1), tuple(cn_1)
        vl_2, cn_2 = n.unique(1)
        self.assertTupleEqual(vl_1, vl_2, f'Unique values not correct. Got {vl_2}. Expected {vl_1}')
        self.assertTupleEqual(cn_1, cn_2, f'Unique counts not correct. Got {cn_2}. Expected {cn_1}')
        with self.assertRaises(en.TensorInstanceException):
            _, _ = n.unique(0)

    def test_shuffle(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.arange(5)
        y = np.arange(5)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        xs, ys = n.shuffle().numpy_lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 1st shuffle')
        n = en.TensorInstanceNumpy((td,), c)
        xs, ys = n.shuffle().numpy_lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 2nd shuffle')

    def test_split_bad(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        length = 20
        width = 5
        a = np.random.rand(length, width)
        b = np.random.rand(length, width)
        c1 = (a, b)
        n1 = en.TensorInstanceNumpy((td,), c1)
        # Split more data than available
        with self.assertRaises(en.TensorInstanceException):
            _ = n1.split_sequential(1, length)

    def test_split_good(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        length = 20
        width = 5
        a = np.random.rand(length, width)
        b = np.random.rand(length, width)
        c1 = (a, b)
        n1 = en.TensorInstanceNumpy((td,), c1)
        test_s, val_s = 2, 1
        train, val, test = n1.split_sequential(val_s, test_s)
        self.assertEqual(len(train.numpy_lists), len(c1), f'Number of train lists changed {len(train.numpy_lists)}')
        self.assertEqual(len(val.numpy_lists), len(c1), f'Number of val lists changed {len(val.numpy_lists)}')
        self.assertEqual(len(test.numpy_lists), len(c1), f'Number of test lists changed {len(test.numpy_lists)}')
        self.assertEqual((train.numpy_lists[0].shape[1]), width, f'Width changed {train.numpy_lists[0].shape[1]}')
        self.assertEqual((train.numpy_lists[1].shape[1]), width, f'Width changed {train.numpy_lists[1].shape[1]}')
        self.assertEqual((val.numpy_lists[0].shape[1]), width, f'Width changed {val.numpy_lists[0].shape[1]}')
        self.assertEqual((val.numpy_lists[1].shape[1]), width, f'Width changed {val.numpy_lists[1].shape[1]}')
        self.assertEqual((test.numpy_lists[0].shape[1]), width, f'Width changed {test.numpy_lists[0].shape[1]}')
        self.assertEqual((test.numpy_lists[1].shape[1]), width, f'Width changed {test.numpy_lists[1].shape[1]}')
        self.assertEqual(len(val), val_s, f'Expected Validation to be size {val_s}. Got {len(val)}')
        self.assertEqual(len(test), test_s, f'Expected Test to be of size {test_s}. Got {len(test)}')
        self.assertEqual(len(train), length-val_s-test_s, f'Unexpected Length training. {length-val_s-test_s}')


class TestLabelIndex(unittest.TestCase):
    """Some tests to see if the can set and get the label indexes"""
    def test_set_label_index_good_single(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.arange(5)
        y = np.arange(5)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        label_ind = 1
        n.label_indexes = label_ind
        self.assertTupleEqual((label_ind,), n.label_indexes, f'Label indexes not equal {n.label_indexes}, {label_ind}')

    def test_set_label_index_good_as_tuple(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.arange(5)
        y = np.arange(5)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        label_ind = 1
        n.label_indexes = (label_ind,)
        self.assertTupleEqual((label_ind,), n.label_indexes, f'Label indexes not equal {n.label_indexes}, {label_ind}')

    def test_set_label_index_not_set(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.arange(5)
        y = np.arange(5)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        # Should fail as the label_indexes were not set.
        with self.assertRaises(en.TensorInstanceException):
            _ = n.label_indexes

    def test_set_label_index_not_in_range(self):
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_FLOAT)
        td = ft.TensorDefinition('td', [f1, f2])
        x = np.arange(5)
        y = np.arange(5)
        c = (x, y)
        n = en.TensorInstanceNumpy((td,), c)
        # Should fail as the label_indexes are not correct.
        with self.assertRaises(en.TensorInstanceException):
            n.label_indexes = -1
        with self.assertRaises(en.TensorInstanceException):
            n.label_indexes = 2
        with self.assertRaises(en.TensorInstanceException):
            n.label_indexes = 'n'


def main():
    unittest.main()


if __name__ == '__main__':
    main()
