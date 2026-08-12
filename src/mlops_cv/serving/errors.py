"""The service's failure vocabulary."""

from __future__ import annotations


class StartupError(RuntimeError):
    """The service cannot be configured; the message says what to run to fix it."""
