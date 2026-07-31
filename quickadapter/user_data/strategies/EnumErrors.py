"""Single source of truth for enum validation error messages.

Dependency-free module (standard library only) so it can be imported by
``Utils`` (which imports from ``LabelTransformer``) and by ``LabelTransformer``
itself without introducing an import cycle.
"""

from collections.abc import Sequence
from typing import Any


def enum_error_message(ctx: str, value: Any, options: Sequence[str]) -> str:
    return f"Invalid {ctx} value {value!r}: supported values are {', '.join(options)}"
