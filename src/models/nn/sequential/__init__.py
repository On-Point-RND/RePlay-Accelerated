from src.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from .sasrec import SasRec
