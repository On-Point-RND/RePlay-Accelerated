"""
Splits data into train and test
"""

from .base_splitter import Splitter, SplitterReturnType
from .cold_user_random_splitter import ColdUserRandomSplitter
from .last_n_splitter import LastNSplitter
from .time_splitter import TimeSplitter

