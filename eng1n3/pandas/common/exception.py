"""
Definition of the Pandas Engine Definition Exception
(c) 2023 tsm
"""


class EnginePandasException(Exception):
    def __init__(self, message: str):
        super().__init__('Error creating source: ' + message)
