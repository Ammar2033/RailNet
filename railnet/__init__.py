"""RailNet — lossless topology-driven execution."""

__version__ = "0.1.0"
__status__ = "research — BF16 PROVEN, hardware PLANNED"

from .compiler import RailNetCompiler
from .dtypes import get_dtype
from .runtime import RailNetDevice, RailNetModel

__all__ = ["RailNetCompiler", "RailNetDevice", "RailNetModel", "__version__", "get_dtype"]
