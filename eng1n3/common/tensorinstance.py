"""
Tensor instance classes, these will go into our NN models.
(c) 2023 3ngin3
"""
import numpy as np
from abc import ABC

from typing import List, Tuple, Any


class TensorInstanceException(Exception):
    def __init__(self, message: str):
        super().__init__("Error Numpy-List: " + message)


class TensorInstance(ABC):
    pass


class TensorInstanceNumpy(TensorInstance):
    """
    Helper Class for a group of numpy arrays. It allows running operations like slicing, sampling, shuffling...
    consistently across a list of numpy arrays.

    Args:
        numpy_list (List[np.ndarray]) : A List of numpy arrays to include in the TensorInstance. Important!
            The numpy lists must all be of the same length. (They must have the same batch dimension)
    """
    def __init__(self, numpy_lists: Tuple[np.ndarray, ...]):
        self._numpy_lists = numpy_lists
        self._val_all_same_0_dim()

    def __getitem__(self, subscript) -> 'TensorInstanceNumpy':
        if isinstance(subscript, slice):
            return self._slice(subscript.start, subscript.stop)
        elif isinstance(subscript, int):
            if subscript < 0:
                subscript += len(self)
            return self._slice(subscript, subscript+1)
        else:
            raise TensorInstanceException(f'Something went wrong. Got wrong subscript type f{subscript}')

    def __len__(self) -> int:
        if len(self.numpy_lists) > 0:
            return len(self.numpy_lists[0])
        else:
            return 0

    def __repr__(self):
        return f'TensorInstance with shapes: {self.shapes}'

    @property
    def numpy_lists(self) -> Tuple[np.ndarray, ...]:
        """
        Property that returns all the numpy ndarray in this TensorInstance

        Returns:
            A list of numpy arrays contained in the TensorInstance
        """
        return self._numpy_lists

    @property
    def number_of_lists(self) -> int:
        """
        Returns the number of numpy arrays contained within this object

        Returns:
            The number of numpy arrays in the list as int object.
        """
        return len(self.numpy_lists)

    @property
    def shapes(self) -> Tuple[Tuple[int], ...]:
        """
        Get the shapes of the underlying numpy lists. Returns a list of Tuples. One tuple for each numpy in the class

        Returns:
            A Tuple of Tuples. Each Tuple contains the shape of a numpy
        """
        return tuple([array.shape for array in self.numpy_lists])

    @property
    def dtype_names(self) -> Tuple[str, ...]:
        """
        Returns the names (i.e. as string) of the dtypes of the underlying numpy arrays.

        Returns:
            Tuple of string dtype string representations
        """
        return tuple([n.dtype.name for n in self.numpy_lists])

    def unique(self, index: int) -> (Tuple[int, ...], Tuple[int, ...]):
        """
        Return the unique sorted entries and counts of a specific array within this TensorInstanceNumpy

        Args:
            index (int): Index of the list for which to run the unique operation.

        Returns:
            A Tuple, the first element is the unique entries, the second entry is the counts.
        """
        self._val_index_in_range(index)
        self._val_is_integer_type(index)
        val, cnt = np.unique(self.numpy_lists[index], return_counts=True)
        return tuple(val), tuple(cnt)

    def shuffle(self) -> 'TensorInstanceNumpy':
        """
        Shuffle the numpy arrays in the list across the 0 (batch) dimension. The shuffling is consistent across lists.
        Meaning that for instance all rows in the various arrays at index x of the input will be moved to index y.
        This will make sure samples are shuffled consistently

        Returns:
            A new TensorInstanceNumpy containing the shuffled numpy arrays.
        """
        permutation = np.random.permutation(self.numpy_lists[0].shape[0])
        shuffled = tuple([n[permutation] for n in self.numpy_lists])
        return TensorInstanceNumpy(shuffled)

    def _slice(self, from_row_number=None, to_row_number=None) -> 'TensorInstanceNumpy':
        """
        Slice all the arrays in this TensorInstance

        Args:
            from_row_number (int): The start number
            to_row_number (int): The end number (exclusive)

        Returns:
            A new instance of TensorInstance with the sliced lists of the input TensorInstance
        """
        if from_row_number is not None and to_row_number is not None:
            self._slice_in_range(from_row_number)
            self._slice_in_range(to_row_number)
            sliced = tuple([n[from_row_number:to_row_number] for n in self.numpy_lists])
        elif from_row_number is not None:
            self._slice_in_range(from_row_number)
            sliced = tuple([n[from_row_number:] for n in self.numpy_lists])
        elif to_row_number is not None:
            self._slice_in_range(to_row_number)
            sliced = tuple([n[:to_row_number] for n in self.numpy_lists])
        else:
            sliced = tuple([n for n in self.numpy_lists])
        return TensorInstanceNumpy(sliced)

    def _val_all_same_0_dim(self):
        """
        Check that all arrays have the shape share in the 0th dimension.

        Raises:
            TensorInstanceException : If the size of the 0th dimension is not the same for all arrays.

        Returns:
            None
        """
        if len(set(list([n_entry.shape[0] for n_entry in self.numpy_lists]))) > 1:
            raise TensorInstanceException(f'All Numpy arrays in a Numpy list must have the same number of rows')

    def _val_index_in_range(self, index: int):
        if index > self.number_of_lists - 1:
            raise TensorInstanceException(
                f'Trying to access index {index} in a numpy list of length {len(self.numpy_lists)}'
            )

    def _slice_in_range(self, index: int):
        if index < 0:
            raise TensorInstanceException(f'Slice index can not be smaller than 0. Got {index}')
        if index > len(self):
            raise TensorInstanceException(
                f'Slice index can not go beyond length of lists. Got {index}, length {len(self.numpy_lists)}'
            )

    def _val_is_integer_type(self, index: int):
        """
        Validation method to check that a specific numpy list in this TensorInstanceNumpy is
        of integer type.

        Args:
            index: The index of the numpy list to check for integer content

        Raises:
             TensorInstanceException: if the content of the numpy list at index {index} is not integer based
        """
        if not np.issubdtype(self.numpy_lists[index].dtype, np.integer):
            raise TensorInstanceException(
                f'List at index <{index}> is not of integer type. That is unexpected'
            )
