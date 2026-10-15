"""Legacy compatibility shim for public release.

The canonical implementation lives under ``reviser.data``.
This top-level package is retained only so older copied modules can still import
``data.*`` paths without forking the implementation.
"""

from reviser.data import *  # noqa: F401,F403
