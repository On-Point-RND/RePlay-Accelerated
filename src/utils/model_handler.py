import functools
import json
import os
import pickle
import warnings
from os.path import join
from pathlib import Path
from typing import Any, Callable, Optional, Union

from src.data.dataset_utils import DatasetLabelEncoder
from src.models import *
from src.splitters import *

from .session_handler import State
from .types import PYSPARK_AVAILABLE


def deprecation_warning(message: Optional[str] = None) -> Callable[..., Any]:
    """
    Decorator that throws deprecation warnings.

    :param message: message to deprecation warning without func name.
    """
    base_msg = "will be deprecated in future versions."

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg = f"{func.__qualname__} {message if message else base_msg}"
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return wrapper

    return decorator
