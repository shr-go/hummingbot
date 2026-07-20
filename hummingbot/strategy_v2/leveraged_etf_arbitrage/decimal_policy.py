"""Deterministic Decimal policy for leveraged-ETF decisions and anchor evidence.

Canonical values have at most 28 coefficient digits, adjusted exponent -18 through
12, and 28 fractional places. Yahoo/checkpoint prices are further limited to 18
fractional places. Tuple metadata and estimated fixed-point length are checked
before formatting or arithmetic; decision arithmetic always uses 28-digit
ROUND_HALF_EVEN semantics in a private local context.
"""

from contextlib import contextmanager
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from typing import Iterator


DECISION_PRECISION = 28
MAX_DECIMAL_COEFFICIENT_DIGITS = 28
MAX_DECIMAL_ADJUSTED_EXPONENT = 12
MIN_DECIMAL_ADJUSTED_EXPONENT = -18
MAX_DECIMAL_SCALE = 28
MAX_DECIMAL_FIXED_LENGTH = 48

YAHOO_MAX_DECIMAL_SCALE = 18
YAHOO_MAX_DECIMAL_FIXED_LENGTH = 32

_DECISION_CONTEXT = Context(
    prec=DECISION_PRECISION,
    rounding=ROUND_HALF_EVEN,
    Emin=-99,
    Emax=99,
)


@contextmanager
def decision_decimal_context() -> Iterator[Context]:
    """Use the one deterministic arithmetic context for all financial outputs."""

    with localcontext(_DECISION_CONTEXT) as context:
        yield context


def validate_bounded_decimal(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    max_scale: int = MAX_DECIMAL_SCALE,
    max_fixed_length: int = MAX_DECIMAL_FIXED_LENGTH,
) -> Decimal:
    """Validate tuple metadata before any operation that can expand fixed-point text."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if value.is_zero() and value.is_signed():
        raise ValueError(f"{field_name} signed zero is outside the canonical Decimal domain")

    sign, digits, exponent = value.as_tuple()
    coefficient_digits = len(digits)
    if coefficient_digits > MAX_DECIMAL_COEFFICIENT_DIGITS:
        raise ValueError(
            f"{field_name} exceeds {MAX_DECIMAL_COEFFICIENT_DIGITS} canonical coefficient digits"
        )
    if not isinstance(exponent, int):
        raise ValueError(f"{field_name} must have a finite integer exponent")
    adjusted_exponent = 0 if value.is_zero() else coefficient_digits + exponent - 1
    if not MIN_DECIMAL_ADJUSTED_EXPONENT <= adjusted_exponent <= MAX_DECIMAL_ADJUSTED_EXPONENT:
        raise ValueError(f"{field_name} adjusted exponent is outside the canonical Decimal domain")
    scale = max(0, -exponent)
    if scale > max_scale:
        raise ValueError(f"{field_name} scale exceeds the canonical Decimal domain")

    if exponent >= 0:
        fixed_length = coefficient_digits + exponent
    elif adjusted_exponent >= 0:
        fixed_length = adjusted_exponent + 1 + 1 + scale
    else:
        fixed_length = 2 + scale
    fixed_length += int(sign)
    if fixed_length > max_fixed_length:
        raise ValueError(f"{field_name} fixed-point length exceeds the canonical Decimal domain")

    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def validate_yahoo_decimal(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    return validate_bounded_decimal(
        value,
        field_name,
        positive=positive,
        nonnegative=nonnegative,
        max_scale=YAHOO_MAX_DECIMAL_SCALE,
        max_fixed_length=YAHOO_MAX_DECIMAL_FIXED_LENGTH,
    )
