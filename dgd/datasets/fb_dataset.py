"""
Compatibility shim for legacy HiGGs Facebook checkpoints.

Some released checkpoints reference the historical module path
`datasets.fb_dataset`, while the current implementation lives in
`datasets.fb_hierarchies`.
"""

from .fb_hierarchies import *  # noqa: F401,F403
