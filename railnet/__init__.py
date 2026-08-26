"""RailNet — lossless topology-driven execution."""

__version__ = "0.1.0"
__status__ = "research — BF16 PROVEN, hardware PLANNED"

from .dtypes import get_dtype
from .compiler import RailNetCompiler
from .runtime import RailNetDevice, RailNetModel

__all__ = ["get_dtype", "RailNetCompiler", "RailNetDevice", "RailNetModel", "__version__"]
