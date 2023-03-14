"""
Unit Tests for TensorInstance class. The data that will go into the models
(c) 2023 tsm
"""
import unittest
import os
import numpy as np

import eng1n3.pandas as en


class TestFeatureBin(unittest.TestCase):
    def test_creation_base(self):
        x = np.arange(10)
        y = np.arange(10)
        c = (x, y)
        n = en.TensorInstanceNumpy(c)
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
        x = np.random.rand(5, 2)
        # This is a smaller numpy. The batch size (first dimension) is smaller.
        y = np.random.rand(2, 2)
        with self.assertRaises(en.TensorInstanceException):
            en.TensorInstanceNumpy((x, y))

    def test_lists(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = (x, y)
        n = en.TensorInstanceNumpy(c)
        self.assertEqual(len(n.numpy_lists), len(c), f'Number of lists does not add up {len(n.numpy_lists)}')
        self.assertEqual((n.numpy_lists[0] == x).all(), True, f'Lists not equal')
        self.assertEqual((n.numpy_lists[1] == y).all(), True, f'Lists not equal')

    def test_slice_good(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = (x, y)
        n = en.TensorInstanceNumpy(c)
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
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = (x, y)
        n = en.TensorInstanceNumpy(c)
        with self.assertRaises(en.TensorInstanceException):
            _ = n[5]
        with self.assertRaises(en.TensorInstanceException):
            _ = n[-6]
        with self.assertRaises(en.TensorInstanceException):
            _ = n[0:6]
        with self.assertRaises(en.TensorInstanceException):
            _ = n[-1:5]

    def test_unique(self):
        x = np.random.rand(5, 2)
        y = np.random.randint(32, size=(5, 2))
        c = (x, y)
        n = en.TensorInstanceNumpy(c)
        vl_1, cn_1 = np.unique(y, return_counts=True)
        vl_1, cn_1 = tuple(vl_1), tuple(cn_1)
        vl_2, cn_2 = n.unique(1)
        self.assertTupleEqual(vl_1, vl_2, f'Unique values not correct. Got {vl_2}. Expected {vl_1}')
        self.assertTupleEqual(cn_1, cn_2, f'Unique counts not correct. Got {cn_2}. Expected {cn_1}')
        with self.assertRaises(en.TensorInstanceException):
            _, _ = n.unique(0)

    def test_shuffle(self):
        x = np.arange(5)
        y = np.arange(5)
        c = (x, y)
        n = en.TensorInstanceNumpy(c)
        xs, ys = n.shuffle().numpy_lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 1st shuffle')
        n = en.TensorInstanceNumpy(c)
        xs, ys = n.shuffle().numpy_lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 2nd shuffle')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
