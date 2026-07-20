"""Deterministic Decimal policy for leveraged-ETF decisions and anchor evidence.

Canonical values have at most 28 coefficient digits, adjusted exponent -18 through
12, and 28 fractional places. Yahoo/checkpoint prices are further limited to 18
fractional places. Tuple metadata and estimated fixed-point length are checked
before formatting or arithmetic; display arithmetic always uses 28-digit
ROUND_HALF_EVEN semantics in a private local context, while boundary decisions
retain exact rational provenance.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Context, Decimal, DecimalException, ROUND_HALF_EVEN, localcontext
from enum import Enum
from fractions import Fraction
from typing import Any, Iterator


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
DECISION_VALUE_SCHEMA_VERSION = 1
DECISION_VALUE_SERIALIZED_FIELDS = (
    "schema_version",
    "certainty",
    "display",
    "exact_numerator",
    "exact_denominator",
    "integrity_hash",
)
_DECISION_VALUE_INTEGER_MAX_LENGTH = 512
_DECISION_VALUE_INTEGER_MAX_BITS = 1701
_DECISION_VALUE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIGNED_CANONICAL_INTEGER_PATTERN = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_POSITIVE_CANONICAL_INTEGER_PATTERN = re.compile(r"^[1-9][0-9]*$")
_DECISION_VALUE_FIELDS = frozenset(DECISION_VALUE_SERIALIZED_FIELDS)


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


class DecisionCertainty(str, Enum):
    EXACT = "EXACT"
    UNTRUSTED_DERIVED = "UNTRUSTED_DERIVED"


class DecisionValueIntegrityError(ValueError):
    """Raised when an explicit strategy decision value cannot be trusted."""


def _canonical_decision_decimal(value: Decimal) -> str:
    validate_bounded_decimal(value, "decision display")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _decision_value_payload(
    *,
    schema_version: int,
    certainty: DecisionCertainty,
    display: Decimal,
    exact_numerator: str | None,
    exact_denominator: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "certainty": certainty.value,
        "display": _canonical_decision_decimal(display),
        "exact_numerator": exact_numerator,
        "exact_denominator": exact_denominator,
    }


def _decision_value_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_decision_integer(value: object, field_name: str, *, positive: bool = False) -> int:
    pattern = _POSITIVE_CANONICAL_INTEGER_PATTERN if positive else _SIGNED_CANONICAL_INTEGER_PATTERN
    if (
        not isinstance(value, str)
        or len(value) > _DECISION_VALUE_INTEGER_MAX_LENGTH
        or pattern.fullmatch(value) is None
    ):
        raise DecisionValueIntegrityError(f"{field_name} must be a bounded canonical integer string")
    return int(value)


def _bounded_fraction_fields(exact_value: Fraction) -> tuple[str, str]:
    for value, field_name in (
        (exact_value.numerator, "exact numerator"),
        (exact_value.denominator, "exact denominator"),
    ):
        if abs(value).bit_length() > _DECISION_VALUE_INTEGER_MAX_BITS:
            raise DecisionValueIntegrityError(f"{field_name} exceeds the bounded exact domain")
    numerator = str(exact_value.numerator)
    denominator = str(exact_value.denominator)
    _parse_decision_integer(numerator, "exact numerator")
    _parse_decision_integer(denominator, "exact denominator", positive=True)
    return numerator, denominator


@dataclass(frozen=True, slots=True, eq=False)
class DecisionValue:
    """Versioned exact or explicitly untrusted value used by strategy decisions.

    Plain ``Decimal`` inputs remain the public contract for caller-supplied raw
    exact values. Derived values use this explicit contract so exact authority is
    retained through arithmetic and serialization. ``UNTRUSTED_DERIVED`` values
    deliberately carry no exact authority and cause strategy decisions to fail
    closed.
    """

    schema_version: int
    certainty: DecisionCertainty
    display: Decimal
    exact_numerator: str | None
    exact_denominator: str | None
    integrity_hash: str

    def __post_init__(self) -> None:
        try:
            self._validate()
        except DecisionValueIntegrityError:
            raise
        except (DecimalException, TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError(str(exception)) from exception

    def _validate(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != DECISION_VALUE_SCHEMA_VERSION:
            raise DecisionValueIntegrityError("decision value schema version must be 1")
        if not isinstance(self.certainty, DecisionCertainty):
            raise DecisionValueIntegrityError("decision certainty is invalid")
        validate_bounded_decimal(self.display, "decision display")

        if self.certainty is DecisionCertainty.EXACT:
            numerator = _parse_decision_integer(self.exact_numerator, "exact numerator")
            denominator = _parse_decision_integer(
                self.exact_denominator,
                "exact denominator",
                positive=True,
            )
            normalized = Fraction(numerator, denominator)
            if (
                str(normalized.numerator) != self.exact_numerator
                or str(normalized.denominator) != self.exact_denominator
            ):
                raise DecisionValueIntegrityError("exact decision fraction must be canonical and reduced")
            if normalized == 0 and not self.display.is_zero():
                raise DecisionValueIntegrityError("exact zero decision must have a zero display")
            if normalized > 0 and self.display <= 0:
                raise DecisionValueIntegrityError("positive exact decision must have a positive display")
            if normalized < 0 and self.display >= 0:
                raise DecisionValueIntegrityError("negative exact decision must have a negative display")
        elif self.exact_numerator is not None or self.exact_denominator is not None:
            raise DecisionValueIntegrityError("untrusted derived value cannot claim exact provenance")

        if not isinstance(self.integrity_hash, str) or _DECISION_VALUE_HASH_PATTERN.fullmatch(
            self.integrity_hash
        ) is None:
            raise DecisionValueIntegrityError("decision value integrity hash must be lowercase SHA-256")
        if self.integrity_hash != _decision_value_hash(self._payload()):
            raise DecisionValueIntegrityError("decision value integrity hash does not match its payload")

    def _payload(self) -> dict[str, Any]:
        return _decision_value_payload(
            schema_version=self.schema_version,
            certainty=self.certainty,
            display=self.display,
            exact_numerator=self.exact_numerator,
            exact_denominator=self.exact_denominator,
        )

    @classmethod
    def from_exact_fraction(
        cls,
        exact_value: Fraction,
        *,
        display: Decimal | None = None,
    ) -> "DecisionValue":
        if not isinstance(exact_value, Fraction):
            raise TypeError("exact decision value must be a Fraction")
        numerator, denominator = _bounded_fraction_fields(exact_value)
        if display is None:
            with decision_decimal_context():
                display = Decimal(exact_value.numerator) / Decimal(exact_value.denominator)
        validate_bounded_decimal(display, "decision display")
        payload = _decision_value_payload(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.EXACT,
            display=display,
            exact_numerator=numerator,
            exact_denominator=denominator,
        )
        return cls(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.EXACT,
            display=display,
            exact_numerator=numerator,
            exact_denominator=denominator,
            integrity_hash=_decision_value_hash(payload),
        )

    @classmethod
    def from_untrusted_derived(cls, display: Decimal) -> "DecisionValue":
        validate_bounded_decimal(display, "untrusted derived display")
        payload = _decision_value_payload(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.UNTRUSTED_DERIVED,
            display=display,
            exact_numerator=None,
            exact_denominator=None,
        )
        return cls(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.UNTRUSTED_DERIVED,
            display=display,
            exact_numerator=None,
            exact_denominator=None,
            integrity_hash=_decision_value_hash(payload),
        )

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> "DecisionValue":
        """Restore the strict version-1 JSON-safe field schema used by adapters."""

        if not isinstance(fields, Mapping) or set(fields) != _DECISION_VALUE_FIELDS:
            raise DecisionValueIntegrityError("decision value fields do not match schema version 1")
        raw_display = fields["display"]
        if not isinstance(raw_display, str) or len(raw_display) > MAX_DECIMAL_FIXED_LENGTH:
            raise DecisionValueIntegrityError("decision display must be a bounded canonical decimal string")
        try:
            display = Decimal(raw_display)
        except DecimalException as exception:
            raise DecisionValueIntegrityError("decision display must be a canonical decimal string") from exception
        if _canonical_decision_decimal(display) != raw_display:
            raise DecisionValueIntegrityError("decision display must be a canonical decimal string")
        try:
            certainty = DecisionCertainty(fields["certainty"])
        except (TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError("decision certainty is invalid") from exception
        try:
            return cls(
                schema_version=fields["schema_version"],
                certainty=certainty,
                display=display,
                exact_numerator=fields["exact_numerator"],
                exact_denominator=fields["exact_denominator"],
                integrity_hash=fields["integrity_hash"],
            )
        except DecisionValueIntegrityError:
            raise
        except (TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError(f"invalid decision value fields: {exception}") from exception

    @property
    def exact_fraction(self) -> Fraction | None:
        self.validate_integrity()
        if self.certainty is DecisionCertainty.UNTRUSTED_DERIVED:
            return None
        assert self.exact_numerator is not None
        assert self.exact_denominator is not None
        return Fraction(int(self.exact_numerator), int(self.exact_denominator))

    def validate_integrity(self) -> None:
        """Revalidate the complete immutable contract at a decision boundary."""

        try:
            self._validate()
        except DecisionValueIntegrityError:
            raise
        except (DecimalException, TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError(str(exception)) from exception

    def verify_integrity_hash(self) -> bool:
        try:
            self.validate_integrity()
        except DecisionValueIntegrityError:
            return False
        return True

    def to_fields(self) -> dict[str, Any]:
        """Return the strict version-1 JSON-safe field schema used by adapters."""

        self.validate_integrity()
        fields = self._payload()
        fields["integrity_hash"] = self.integrity_hash
        return fields

    def _operand(self, other: object) -> tuple[Decimal, Fraction | None] | None:
        if isinstance(other, DecisionValue):
            return other.display, other.exact_fraction
        if isinstance(other, Decimal):
            validate_bounded_decimal(other, "decision arithmetic operand")
            return other, Fraction(other)
        return None

    def _binary(
        self,
        other: object,
        operation: str,
        *,
        reflected: bool = False,
    ) -> "DecisionValue":
        operand = self._operand(other)
        if operand is None:
            return NotImplemented
        other_display, other_exact = operand
        left_display, right_display = (
            (other_display, self.display) if reflected else (self.display, other_display)
        )
        left_exact, right_exact = (
            (other_exact, self.exact_fraction) if reflected else (self.exact_fraction, other_exact)
        )
        with decision_decimal_context():
            if operation == "add":
                display = left_display + right_display
                exact = None if left_exact is None or right_exact is None else left_exact + right_exact
            else:
                display = left_display - right_display
                exact = None if left_exact is None or right_exact is None else left_exact - right_exact
        if display.is_zero():
            display = display.copy_abs()
        validate_bounded_decimal(display, "decision arithmetic result")
        if exact is None:
            return DecisionValue.from_untrusted_derived(display)
        return DecisionValue.from_exact_fraction(exact, display=display)

    def __add__(self, other: object) -> "DecisionValue":
        return self._binary(other, "add")

    def __radd__(self, other: object) -> "DecisionValue":
        return self._binary(other, "add", reflected=True)

    def __sub__(self, other: object) -> "DecisionValue":
        return self._binary(other, "subtract")

    def __rsub__(self, other: object) -> "DecisionValue":
        return self._binary(other, "subtract", reflected=True)

    def __pos__(self) -> "DecisionValue":
        return self

    def __neg__(self) -> "DecisionValue":
        exact = self.exact_fraction
        with decision_decimal_context():
            display = -self.display
        if display.is_zero():
            display = display.copy_abs()
        if exact is None:
            return DecisionValue.from_untrusted_derived(display)
        return DecisionValue.from_exact_fraction(-exact, display=display)

    def __abs__(self) -> "DecisionValue":
        exact = self.exact_fraction
        with decision_decimal_context():
            display = abs(self.display)
        if exact is None:
            return DecisionValue.from_untrusted_derived(display)
        return DecisionValue.from_exact_fraction(abs(exact), display=display)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DecisionValue):
            return (
                self.certainty is other.certainty
                and self.display == other.display
                and self.exact_fraction == other.exact_fraction
            )
        if isinstance(other, Decimal):
            return self.display == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.display)

    def _comparison_value(self) -> Fraction:
        exact = self.exact_fraction
        if exact is None:
            raise DecisionValueIntegrityError(
                "untrusted derived decision value cannot authorize an ordered comparison"
            )
        return exact

    def _comparison_operand(self, other: object) -> Fraction | None:
        if isinstance(other, DecisionValue):
            return other._comparison_value()
        if isinstance(other, Decimal):
            return Fraction(other)
        return None

    def __lt__(self, other: object):
        operand = self._comparison_operand(other)
        return NotImplemented if operand is None else self._comparison_value() < operand

    def __le__(self, other: object):
        operand = self._comparison_operand(other)
        return NotImplemented if operand is None else self._comparison_value() <= operand

    def __gt__(self, other: object):
        operand = self._comparison_operand(other)
        return NotImplemented if operand is None else self._comparison_value() > operand

    def __ge__(self, other: object):
        operand = self._comparison_operand(other)
        return NotImplemented if operand is None else self._comparison_value() >= operand

    def as_integer_ratio(self) -> tuple[int, int]:
        exact = self.exact_fraction
        if exact is None:
            raise ValueError("untrusted derived decision value has no exact ratio")
        return exact.numerator, exact.denominator

    def __str__(self) -> str:
        return str(self.display)

    def __format__(self, format_spec: str) -> str:
        return format(self.display, format_spec)
