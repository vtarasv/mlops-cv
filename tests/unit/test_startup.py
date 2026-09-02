"""The startup guard: a refusal to start is an exit code and the fix, never a traceback."""

from __future__ import annotations

import logging

import pytest

from mlops_cv.startup import EXIT_CODE, StartupError, exits_on_startup_error


def test_a_startup_error_becomes_the_exit_code_and_a_logged_fix(caplog) -> None:
    @exits_on_startup_error
    def main(argv: list[str] | None = None) -> int:
        raise StartupError("no champion — run `make train`")

    with caplog.at_level(logging.ERROR):
        assert main([]) == EXIT_CODE == 2
    assert "make train" in caplog.text


def test_a_clean_start_returns_what_main_returns() -> None:
    @exits_on_startup_error
    def main(argv: list[str] | None = None) -> int:
        return 0

    assert main([]) == 0


def test_other_failures_are_not_swallowed() -> None:
    """Only the named operational states are exit codes; a bug still stops the process loudly."""

    @exits_on_startup_error
    def main() -> int:
        raise ValueError("a bug")

    with pytest.raises(ValueError, match="a bug"):
        main()


def test_the_refusal_is_logged_under_the_entrypoints_own_logger(caplog) -> None:
    """Container logs attribute the line to the process that refused, not to the guard."""

    @exits_on_startup_error
    def main() -> int:
        raise StartupError("refused")

    with caplog.at_level(logging.ERROR):
        main()
    assert [record.name for record in caplog.records] == [__name__]
