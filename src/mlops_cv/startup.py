"""Refusing to start."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

# The exit code every entrypoint uses for "cannot start; the log line says what to run".
EXIT_CODE = 2


class StartupError(RuntimeError):
    """The process cannot start; the message says what to run to fix it."""


def exits_on_startup_error[**P](main: Callable[P, int]) -> Callable[P, int]:
    """Wrap an entrypoint's ``main``: a :class:`StartupError` is logged and becomes EXIT_CODE."""
    logger = logging.getLogger(main.__module__)

    @functools.wraps(main)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> int:
        try:
            return main(*args, **kwargs)
        except StartupError as exc:
            logger.error(str(exc))
            return EXIT_CODE

    return guarded
