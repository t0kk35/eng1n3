"""
Tensor instance classes, these will go into our NN models.
(c) 2023 3ngin3
"""
import numpy as np
from abc import ABC, abstractmethod

from f3atur3s import TensorDefinition, LearningCategory, FeatureLabel

from typing import List, Tuple, Any


class TensorInstanceException(Exception):
    def __init__(self, message: str):
        super().__init__("Error TensorInstance: " + message)


class TensorInstance(ABC):
    @property
    @abstractmethod
    def target_tensor_def(self) -> Tuple[TensorDefinition, ...]:
        pass

    @property
    @abstractmethod
    def label_indexes(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def number_of_lists(self) -> int:
        pass

    @property
    @abstractmethod
    def learning_categories(self) -> Tuple[LearningCategory, ...]:
        pass

    def val_indexes_in_range(self, indexes: Tuple[int, ...]):
        for index in indexes:
            if index < 0:
                raise TensorInstanceException(
                    f'Indexes can not be negative. Index as {index}'
                )
            elif index > self.number_of_lists - 1:
                raise TensorInstanceException(
                    f'Index <{index}> out of range, this instance only has {self.number_of_lists} entries'
                )


class TensorInstanceNumpy(TensorInstance):
    """
    Helper Class for a group of numpy arrays. It allows running operations like slicing, sampling, shuffling...
    consistently across a list of numpy arrays.

    Args:
        numpy_lists (List[np.ndarray]) : A List of numpy arrays to include in the TensorInstance. Important!
            The numpy lists must all be of the same length. (They must have the same batch dimension)
    """
    def __init__(self, target_tensor_def: Tuple[TensorDefinition, ...], numpy_lists: Tuple[np.ndarray, ...]):
        self._label_indexes: Tuple[int, ...] = self._find_label_indexes(target_tensor_def)
        self._target_tensor_def = target_tensor_def
        self._numpy_lists = numpy_lists
        self._learning_categories = self._get_learning_categories()
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
    def label_indexes(self) -> Tuple[int, ...]:
        if len(self._label_indexes) != 0:
            return self._label_indexes
        else:
            raise TensorInstanceException(
                f'The label_index property has not been set. This most likely means the TensorInstance does not ' +
                f'contain a TensorDefinition with "FeatureLabel" class features in it'
            )

    @property
    def target_tensor_def(self) -> Tuple[TensorDefinition, ...]:
        return self._target_tensor_def

    @property
    def non_label_target_tensor_def(self) -> Tuple[TensorDefinition, ...]:
        """
        Property that return all Target TensorDefinition used to build this TensorInstance, except for the
        TensorDefinition(s) that contain the label information.
        """
        return tuple(td for i, td in enumerate(self.target_tensor_def) if i not in self.label_indexes)

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

    @property
    def learning_categories(self) -> Tuple[LearningCategory, ...]:
        return self._learning_categories

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
        return TensorInstanceNumpy(self.target_tensor_def, shuffled)

    def split_sequential(self,
                         val_number: int,
                         test_number: int
                         ) -> Tuple['TensorInstanceNumpy', 'TensorInstanceNumpy', 'TensorInstanceNumpy']:
        """
        Split a numpy list into training, validation and test. This selects purely sequentially, it does not order
        so if the order of some 'time' aspect is important, then make sure to order the list first!
        The first portion of the data is training, the middle is the validation and the end of the data is the test.
        This is almost always the best way to split transactional data. First the 'test_number' of records data is
        taken from the end of the arrays. Of what is left the 'val_number' is taken all that is left is training.

        Args:
            val_number (int): Number of records to allocate to the validation set
            test_number (int): Number of records to allocate to the test set.

        Returns:
            Tuple of 3 TensorInstance objects containing the training, validation and test data respectively
        """
        self._val_val_plus_test_smaller_than_length(val_number, test_number)
        # Take x from end of lists as test
        test = self._slice(from_row_number=len(self)-test_number, to_row_number=len(self))
        self._copy_across_properties(self, test)
        # Take another x from what is left at the end and not in test
        val = self._slice(from_row_number=len(self)-test_number-val_number, to_row_number=len(self)-test_number)
        self._copy_across_properties(self, val)
        # Take rest
        train = self._slice(to_row_number=len(self)-test_number-val_number)
        self._copy_across_properties(self, train)
        return train, val, test

    @staticmethod
    def _find_label_indexes(target_tensor_def: Tuple[TensorDefinition, ...]) -> Tuple[int, ...]:
        """
        Local method that tries to find out what indexes are the labels in the data set (if any). Label
        TensorDefinitions are TensorDefinitions that only contain feature of type 'FeatureLabel'.

        Args:
             target_tensor_def: A tuple of TensorDefinitions to check

        Returns:
            A tuple of ints of (potentially empty) that are the indexes of the lists that hold the labels.
        """
        ind = [i for i, td in enumerate(target_tensor_def) if all([isinstance(f, FeatureLabel) for f in td.features])]
        return tuple(ind)

    @staticmethod
    def _copy_across_properties(old_ti: 'TensorInstanceNumpy', new_ti: 'TensorInstanceNumpy') -> None:
        """
        Small helper method to copy across some properties from the old to a new TensorDefinitionNumpy. We'll
        use this when we create a new instance from an old one.

        Args:
            old_ti: The existing TensorDefinitionNupy.
            new_ti: The new target TensorDefinitionNumpy into which we want to copy the properties from the old instance
        """
        new_ti._label_indexes = old_ti._label_indexes

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
        return TensorInstanceNumpy(self.target_tensor_def, sliced)

    def filter_label(self, label: Any) -> 'TensorInstanceNumpy':
        """
        Method to filter a specific class from the labels. It can for instance be used to filter Fraud or Non-Fraud
        Args:
            label (Any): The label value (class) we want to filter.
        Returns:
            New filtered numpy list, filtered on the label value
        """
        label_index = self._get_single_label()
        labels = self.numpy_lists[label_index]
        if len(labels.shape) == 2:
            labels = np.squeeze(labels)
        index = np.where(labels == label)
        lists = tuple(npl[index] for npl in self.numpy_lists)
        return TensorInstanceNumpy(self.target_tensor_def, lists)

    def _get_learning_categories(self) -> Tuple[LearningCategory, ...]:
        lcs: List[LearningCategory] = []
        for td in self.target_tensor_def:
            lc = list(set([f.learning_category for f in td.features]))
            if len(lc) > 1:
                raise TensorInstanceException(
                    f'TensorDefinition {td.name} contained more than one Learning category. Found {lc}. Make sure ' +
                    f'each TensorDefinition has a single unique Learning Category'
                )
            else:
                lcs.extend(lc)
        return tuple(lcs)

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
                f'Slice index can not go beyond length of lists. Got {index}, length {len(self)}'
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

    def _val_val_plus_test_smaller_than_length(self, validation: int, test: int):
        if validation + test >= len(self):
            raise TensorInstanceException(
                f'The number of validation <{validation}> + the number of test <{test}> records. Is bigger than the ' +
                f'Length of the Numpy List <{len(self)}> '
            )

    def _get_single_label(self) -> int:
        if len(self.label_indexes) < 1:
            raise TensorInstanceException(
                f'One of the Tensor Definitions {[td.name for td in self.target_tensor_def]} should have a label ' +
                f'feature'
            )
        elif len(self._label_indexes) > 1:
            raise TensorInstanceException(
                f'Operation is only possible if the TensorInstance has a single label TensorDefinition'
            )
        else:
            return self.label_indexes[0]
