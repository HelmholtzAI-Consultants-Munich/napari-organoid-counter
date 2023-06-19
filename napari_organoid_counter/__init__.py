try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import OrganoidCounterWidget
from ._reader import reader_function, get_reader
