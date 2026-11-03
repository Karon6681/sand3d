"""Parts that are tedious to build by hand, as verbs: a gear, a knurled grip, a thread."""
from .turned import knurl, thread
from .gear import gear

__all__ = ["gear", "knurl", "thread"]
