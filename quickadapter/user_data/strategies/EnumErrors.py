"""Canonical enum validation error message. Dependency-free (stdlib only).

Owns the ``Invalid <ctx> value <value>: supported values are <options>`` form;
messages that deviate from it (custom prefix/infix/suffix) are built inline.
"""

from collections.abc import Sequence
from typing import Any


def enum_error_message(ctx: str, value: Any, options: Sequence[str]) -> str:
    return f"Invalid {ctx} value {value!r}: supported values are {', '.join(options)}"
