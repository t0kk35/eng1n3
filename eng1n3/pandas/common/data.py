
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Type, Union
from f3atur3s import Feature

from .exception import EnginePandasException

# Panda types dictionary. Values are Tuples, the first entry in the type used for reading, the second for converting
PandaTypes: Dict[str, Tuple[str, Type]] = {
    'STRING': ('object', object),
    'FLOAT': ('float64', np.float64),
    'FLOAT_32': ('float32', np.float32),
    'INTEGER': ('int32', np.int32),
    'INT_8': ('int8', np.int8),
    'INT_16': ('int16', np.int16),
    'INT_64': ('int64', np.int64),
    'DATE': ('str', pd.Timestamp),
    'CATEGORICAL': ('category', pd.Categorical),
    'DATETIME': ('str', pd.Timestamp)
}


def pandas_type(feature: Feature, default: str = None, read: bool = True) -> Union[str, Type]:
    """
    Helper function that determines the panda (and numpy) data types for a specific feature. Base on the f_type

    Args:
        feature: (Feature) A feature definition
        default: (string) A default data type
        read: (bool). Flag indicating whether to return the 'read' type or the 'interpret' type.
        Default is 'read'
    Returns:
        (Union[str, Type] Panda (Numpy) data type as a string or object depending on the read parameter
    """
    if feature is None:
        return default
    panda_type = PandaTypes.get(feature.type.name, (default, default))
    if panda_type is None:
        raise EnginePandasException(f'Did not find pandas type for {feature.name}')
    else:
        if read:
            return panda_type[0]
        else:
            return panda_type[1]
