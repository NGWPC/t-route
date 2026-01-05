# Error Warning and Trapping System Package API
# ewts/__init__.py


from .constants import MODULE_NAME
from .config import configure_logging

__all__ = ["MODULE_NAME", "configure_logging"]
