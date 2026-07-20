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

from pydantic import StrictInt


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
DECISION_VALUE_SCHEMA_VERSION = 3
DECISION_VALUE_SERIALIZED_FIELDS = (
    "schema_version",
    "certainty",
    "semantic_kind",
    "operands",
    "display",
    "exact_numerator",
    "exact_denominator",
    "integrity_hash",
)
RAW_EXACT_DECISION_SCHEMA_VERSION = 1
RAW_EXACT_DECISION_SERIALIZED_FIELDS = (
    "schema_version",
    "kind",
    "value",
)
_DECISION_VALUE_INTEGER_MAX_LENGTH = 512
_DECISION_VALUE_INTEGER_MAX_BITS = 1701
_DECISION_VALUE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIGNED_CANONICAL_INTEGER_PATTERN = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_POSITIVE_CANONICAL_INTEGER_PATTERN = re.compile(r"^[1-9][0-9]*$")
_DECISION_VALUE_FIELDS = frozenset(DECISION_VALUE_SERIALIZED_FIELDS)
_RAW_EXACT_DECISION_FIELDS = frozenset(RAW_EXACT_DECISION_SERIALIZED_FIELDS)


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
    EXACT_DERIVED = "EXACT_DERIVED"
    UNTRUSTED_DERIVED = "UNTRUSTED_DERIVED"


class DecisionSemanticKind(str, Enum):
    THEORETICAL_ETF_PRICE = "THEORETICAL_ETF_PRICE"
    OPPORTUNITY_NET_BP = "OPPORTUNITY_NET_BP"


class RawDecisionKind(str, Enum):
    ETF_PRICE = "ETF_PRICE"
    NET_BP = "NET_BP"


class DecisionValueIntegrityError(ValueError):
    """Raised when a strategy decision contract cannot be independently verified."""


def _canonical_decision_decimal(value: Decimal) -> str:
    validate_bounded_decimal(value, "decision Decimal")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _canonicalized_decision_decimal(value: Decimal) -> Decimal:
    return Decimal(_canonical_decision_decimal(value))


def _parse_canonical_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, str) or len(value) > MAX_DECIMAL_FIXED_LENGTH:
        raise DecisionValueIntegrityError(f"{field_name} must be a bounded canonical decimal string")
    try:
        parsed = Decimal(value)
    except DecimalException as exception:
        raise DecisionValueIntegrityError(
            f"{field_name} must be a bounded canonical decimal string"
        ) from exception
    try:
        canonical = _canonical_decision_decimal(parsed)
    except (TypeError, ValueError) as exception:
        raise DecisionValueIntegrityError(str(exception)) from exception
    if canonical != value:
        raise DecisionValueIntegrityError(f"{field_name} must be a canonical decimal string")
    return parsed


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


@dataclass(frozen=True, slots=True)
class RawExactDecision:
    """An explicit exact raw value whose intended decision use is kind-bound."""

    schema_version: StrictInt
    kind: RawDecisionKind
    value: Decimal

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != RAW_EXACT_DECISION_SCHEMA_VERSION:
            raise DecisionValueIntegrityError("raw exact decision schema version must be 1")
        if not isinstance(self.kind, RawDecisionKind):
            raise DecisionValueIntegrityError("raw exact decision kind is invalid")
        try:
            validate_bounded_decimal(
                self.value,
                "raw exact decision value",
                positive=self.kind is RawDecisionKind.ETF_PRICE,
            )
        except (TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError(str(exception)) from exception

    @classmethod
    def etf_price(cls, value: Decimal) -> "RawExactDecision":
        validate_bounded_decimal(value, "raw ETF price", positive=True)
        return cls(
            schema_version=RAW_EXACT_DECISION_SCHEMA_VERSION,
            kind=RawDecisionKind.ETF_PRICE,
            value=value,
        )

    @classmethod
    def net_bp(cls, value: Decimal) -> "RawExactDecision":
        validate_bounded_decimal(value, "raw net bp")
        return cls(
            schema_version=RAW_EXACT_DECISION_SCHEMA_VERSION,
            kind=RawDecisionKind.NET_BP,
            value=value,
        )

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> "RawExactDecision":
        if not isinstance(fields, Mapping) or set(fields) != _RAW_EXACT_DECISION_FIELDS:
            raise DecisionValueIntegrityError("raw exact decision fields do not match schema version 1")
        try:
            kind = RawDecisionKind(fields["kind"])
        except (TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError("raw exact decision kind is invalid") from exception
        return cls(
            schema_version=fields["schema_version"],
            kind=kind,
            value=_parse_canonical_decimal(fields["value"], "raw exact decision value"),
        )

    @property
    def exact_fraction(self) -> Fraction:
        return Fraction(self.value)

    def validate_integrity(self) -> None:
        self.__post_init__()

    def to_fields(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "value": _canonical_decision_decimal(self.value),
        }


_THEORETICAL_OPERAND_SPEC = (
    ("stock_price", "positive"),
    ("stock_anchor", "positive"),
    ("etf_anchor", "positive"),
    ("etf_daily_multiplier", "positive"),
)
_NET_BP_OPERAND_SPEC = (
    ("stock_anchor", "positive"),
    ("etf_anchor", "positive"),
    ("etf_daily_multiplier", "positive"),
    ("stock_entry_price", "positive"),
    ("etf_entry_price", "positive"),
    ("etf_quantity", "positive"),
    ("stock_quantity", "positive"),
    ("stock_contract_multiplier", "positive"),
    ("etf_contract_multiplier", "positive"),
    ("maker_fee_bp", "nonnegative"),
    ("taker_fee_bp", "nonnegative"),
    ("maker_slippage_bp_per_fill", "nonnegative"),
)
_OPERAND_SPEC_BY_KIND = {
    DecisionSemanticKind.THEORETICAL_ETF_PRICE: _THEORETICAL_OPERAND_SPEC,
    DecisionSemanticKind.OPPORTUNITY_NET_BP: _NET_BP_OPERAND_SPEC,
}
_BASIS_POINTS = Decimal("10000")


def _operand_spec(kind: DecisionSemanticKind) -> tuple[tuple[str, str], ...]:
    try:
        return _OPERAND_SPEC_BY_KIND[kind]
    except KeyError as exception:
        raise DecisionValueIntegrityError("derived decision semantic kind is unsupported") from exception


def _validate_operand_decimal(value: object, name: str, requirement: str) -> Decimal:
    try:
        return validate_bounded_decimal(
            value,
            f"decision operand {name}",
            positive=requirement == "positive",
            nonnegative=requirement == "nonnegative",
        )
    except (TypeError, ValueError) as exception:
        raise DecisionValueIntegrityError(str(exception)) from exception


def _canonical_operand_pairs(
    kind: DecisionSemanticKind,
    operands: Mapping[str, Decimal],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(operands, Mapping):
        raise DecisionValueIntegrityError("derived decision operands must be a mapping")
    spec = _operand_spec(kind)
    expected_names = {name for name, _ in spec}
    if set(operands) != expected_names:
        raise DecisionValueIntegrityError("derived decision operands do not match semantic kind")
    return tuple(
        (
            name,
            _canonical_decision_decimal(
                _validate_operand_decimal(operands[name], name, requirement)
            ),
        )
        for name, requirement in spec
    )


def _operand_values(
    kind: DecisionSemanticKind,
    operands: object,
) -> dict[str, Decimal]:
    spec = _operand_spec(kind)
    if not isinstance(operands, tuple) or len(operands) != len(spec):
        raise DecisionValueIntegrityError("derived decision operands do not match semantic kind")
    parsed: dict[str, Decimal] = {}
    for pair, (expected_name, requirement) in zip(operands, spec):
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or pair[0] != expected_name
            or not isinstance(pair[1], str)
        ):
            raise DecisionValueIntegrityError("derived decision operands do not match semantic kind")
        value = _parse_canonical_decimal(pair[1], f"decision operand {expected_name}")
        parsed[expected_name] = _validate_operand_decimal(value, expected_name, requirement)
    return parsed


def _exact_theoretical_from_values(values: Mapping[str, Decimal]) -> Fraction:
    stock_price = Fraction(values["stock_price"])
    stock_anchor = Fraction(values["stock_anchor"])
    return Fraction(values["etf_anchor"]) * (
        1
        + Fraction(values["etf_daily_multiplier"])
        * (stock_price / stock_anchor - 1)
    )


def _display_fraction(exact_value: Fraction, field_name: str) -> Decimal:
    with decision_decimal_context():
        display = Decimal(exact_value.numerator) / Decimal(exact_value.denominator)
    if display.is_zero():
        display = display.copy_abs()
    try:
        return _canonicalized_decision_decimal(
            validate_bounded_decimal(display, field_name)
        )
    except (TypeError, ValueError) as exception:
        raise DecisionValueIntegrityError(str(exception)) from exception


def _exact_net_bp_from_values(values: Mapping[str, Decimal]) -> tuple[Fraction, Fraction, Fraction]:
    stock_anchor = Fraction(values["stock_anchor"])
    etf_anchor = Fraction(values["etf_anchor"])
    multiplier = Fraction(values["etf_daily_multiplier"])
    stock_price = Fraction(values["stock_entry_price"])
    etf_price = Fraction(values["etf_entry_price"])
    etf_quantity = Fraction(values["etf_quantity"])
    stock_contract = Fraction(values["stock_contract_multiplier"])
    etf_contract = Fraction(values["etf_contract_multiplier"])
    exact_theoretical = etf_anchor * (
        1 + multiplier * (stock_price / stock_anchor - 1)
    )
    exact_stock_quantity = Fraction(values["stock_quantity"])
    exact_etf_notional = etf_quantity * etf_contract * etf_price
    exact_stock_notional = exact_stock_quantity * stock_contract * stock_price
    exact_gross_notional = exact_etf_notional + exact_stock_notional
    exact_gross_profit = abs(etf_price - exact_theoretical) * etf_quantity * etf_contract
    exact_raw_bp = Fraction(_BASIS_POINTS) * exact_gross_profit / exact_gross_notional
    exact_total_quote = 2 * (
        exact_etf_notional
        * Fraction(values["maker_fee_bp"])
        / Fraction(_BASIS_POINTS)
        + exact_stock_notional
        * Fraction(values["taker_fee_bp"])
        / Fraction(_BASIS_POINTS)
        + exact_etf_notional
        * Fraction(values["maker_slippage_bp_per_fill"])
        / Fraction(_BASIS_POINTS)
    )
    exact_cost_bp = Fraction(_BASIS_POINTS) * exact_total_quote / exact_gross_notional
    return exact_gross_profit, exact_raw_bp, exact_raw_bp - exact_cost_bp


def _net_bp_display_from_values(
    values: Mapping[str, Decimal],
    exact_gross_profit: Fraction,
) -> Decimal:
    with decision_decimal_context():
        etf_notional = (
            values["etf_quantity"]
            * values["etf_contract_multiplier"]
            * values["etf_entry_price"]
        )
        stock_notional = (
            values["stock_quantity"]
            * values["stock_contract_multiplier"]
            * values["stock_entry_price"]
        )
        gross_notional = etf_notional + stock_notional
        gross_profit = (
            Decimal(exact_gross_profit.numerator)
            / Decimal(exact_gross_profit.denominator)
        )
        raw_bp = _BASIS_POINTS * gross_profit / gross_notional
        etf_maker_fee = etf_notional * values["maker_fee_bp"] / _BASIS_POINTS
        stock_taker_fee = stock_notional * values["taker_fee_bp"] / _BASIS_POINTS
        etf_slippage = (
            etf_notional
            * values["maker_slippage_bp_per_fill"]
            / _BASIS_POINTS
        )
        total_quote = Decimal("2") * (etf_maker_fee + stock_taker_fee + etf_slippage)
        total_bp = _BASIS_POINTS * total_quote / gross_notional
        display = raw_bp - total_bp
    if display.is_zero():
        display = display.copy_abs()
    try:
        return _canonicalized_decision_decimal(
            validate_bounded_decimal(display, "opportunity net bp display")
        )
    except (TypeError, ValueError) as exception:
        raise DecisionValueIntegrityError(str(exception)) from exception


def _recompute_decision(
    kind: DecisionSemanticKind,
    operands: tuple[tuple[str, str], ...],
) -> tuple[Fraction, Decimal]:
    values = _operand_values(kind, operands)
    if kind is DecisionSemanticKind.THEORETICAL_ETF_PRICE:
        exact = _exact_theoretical_from_values(values)
        if exact <= 0:
            raise DecisionValueIntegrityError("theoretical ETF price must be positive")
        return exact, _display_fraction(exact, "theoretical ETF price display")
    if kind is DecisionSemanticKind.OPPORTUNITY_NET_BP:
        exact_theoretical = Fraction(values["etf_anchor"]) * (
            1
            + Fraction(values["etf_daily_multiplier"])
            * (
                Fraction(values["stock_entry_price"])
                / Fraction(values["stock_anchor"])
                - 1
            )
        )
        if exact_theoretical <= 0:
            raise DecisionValueIntegrityError(
                "opportunity operands produce a nonpositive theoretical ETF price"
            )
        exact_gross_profit, _, exact = _exact_net_bp_from_values(values)
        if exact_gross_profit <= 0:
            raise DecisionValueIntegrityError("opportunity operands contain no directional spread")
        return exact, _net_bp_display_from_values(values, exact_gross_profit)
    raise DecisionValueIntegrityError("derived decision semantic kind is unsupported")


def _decision_value_payload(
    *,
    schema_version: int,
    certainty: DecisionCertainty,
    semantic_kind: DecisionSemanticKind | None,
    operands: tuple[tuple[str, str], ...],
    display: Decimal,
    exact_numerator: str | None,
    exact_denominator: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "certainty": certainty.value,
        "semantic_kind": None if semantic_kind is None else semantic_kind.value,
        "operands": [[name, value] for name, value in operands],
        "display": _canonical_decision_decimal(display),
        "exact_numerator": exact_numerator,
        "exact_denominator": exact_denominator,
    }


@dataclass(frozen=True, slots=True, eq=False)
class DecisionValue:
    """A derived decision whose authority is recomputed from fixed semantics."""

    schema_version: StrictInt
    certainty: DecisionCertainty
    semantic_kind: DecisionSemanticKind | None
    operands: tuple[tuple[str, str], ...]
    display: Decimal
    exact_numerator: str | None
    exact_denominator: str | None
    integrity_hash: str

    def __post_init__(self) -> None:
        self.validate_integrity()

    def _validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != DECISION_VALUE_SCHEMA_VERSION:
            raise DecisionValueIntegrityError("decision value schema version must be 3")
        if not isinstance(self.certainty, DecisionCertainty):
            raise DecisionValueIntegrityError("decision certainty is invalid")
        try:
            validate_bounded_decimal(self.display, "decision display")
        except (TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError(str(exception)) from exception

        if self.certainty is DecisionCertainty.EXACT_DERIVED:
            if not isinstance(self.semantic_kind, DecisionSemanticKind):
                raise DecisionValueIntegrityError("exact derived decision requires a semantic kind")
            recomputed_exact, recomputed_display = _recompute_decision(
                self.semantic_kind,
                self.operands,
            )
            expected_numerator, expected_denominator = _bounded_fraction_fields(recomputed_exact)
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
            if (
                self.exact_numerator != expected_numerator
                or self.exact_denominator != expected_denominator
            ):
                raise DecisionValueIntegrityError(
                    "exact decision fraction does not match recomputed semantic operands"
                )
            if self.display.as_tuple() != recomputed_display.as_tuple():
                raise DecisionValueIntegrityError(
                    "decision display does not match recomputed semantic operands"
                )
        else:
            if (
                self.semantic_kind is not None
                or self.operands != ()
                or self.exact_numerator is not None
                or self.exact_denominator is not None
            ):
                raise DecisionValueIntegrityError(
                    "untrusted derived value cannot claim semantic exact authority"
                )

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
            semantic_kind=self.semantic_kind,
            operands=self.operands,
            display=self.display,
            exact_numerator=self.exact_numerator,
            exact_denominator=self.exact_denominator,
        )

    @classmethod
    def _from_semantic_operands(
        cls,
        kind: DecisionSemanticKind,
        operands: Mapping[str, Decimal],
    ) -> "DecisionValue":
        canonical_operands = _canonical_operand_pairs(kind, operands)
        exact, display = _recompute_decision(kind, canonical_operands)
        numerator, denominator = _bounded_fraction_fields(exact)
        payload = _decision_value_payload(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.EXACT_DERIVED,
            semantic_kind=kind,
            operands=canonical_operands,
            display=display,
            exact_numerator=numerator,
            exact_denominator=denominator,
        )
        return cls(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.EXACT_DERIVED,
            semantic_kind=kind,
            operands=canonical_operands,
            display=display,
            exact_numerator=numerator,
            exact_denominator=denominator,
            integrity_hash=_decision_value_hash(payload),
        )

    @classmethod
    def from_theoretical_price_operands(
        cls,
        *,
        stock_price: Decimal,
        stock_anchor: Decimal,
        etf_anchor: Decimal,
        etf_daily_multiplier: Decimal,
    ) -> "DecisionValue":
        return cls._from_semantic_operands(
            DecisionSemanticKind.THEORETICAL_ETF_PRICE,
            {
                "stock_price": stock_price,
                "stock_anchor": stock_anchor,
                "etf_anchor": etf_anchor,
                "etf_daily_multiplier": etf_daily_multiplier,
            },
        )

    @classmethod
    def from_opportunity_net_bp_operands(
        cls,
        *,
        stock_anchor: Decimal,
        etf_anchor: Decimal,
        etf_daily_multiplier: Decimal,
        stock_entry_price: Decimal,
        etf_entry_price: Decimal,
        etf_quantity: Decimal,
        stock_quantity: Decimal,
        stock_contract_multiplier: Decimal,
        etf_contract_multiplier: Decimal,
        maker_fee_bp: Decimal,
        taker_fee_bp: Decimal,
        maker_slippage_bp_per_fill: Decimal,
    ) -> "DecisionValue":
        return cls._from_semantic_operands(
            DecisionSemanticKind.OPPORTUNITY_NET_BP,
            {
                "stock_anchor": stock_anchor,
                "etf_anchor": etf_anchor,
                "etf_daily_multiplier": etf_daily_multiplier,
                "stock_entry_price": stock_entry_price,
                "etf_entry_price": etf_entry_price,
                "etf_quantity": etf_quantity,
                "stock_quantity": stock_quantity,
                "stock_contract_multiplier": stock_contract_multiplier,
                "etf_contract_multiplier": etf_contract_multiplier,
                "maker_fee_bp": maker_fee_bp,
                "taker_fee_bp": taker_fee_bp,
                "maker_slippage_bp_per_fill": maker_slippage_bp_per_fill,
            },
        )

    @classmethod
    def from_exact_fraction(
        cls,
        exact_value: Fraction,
        *,
        display: Decimal | None = None,
    ) -> "DecisionValue":
        raise DecisionValueIntegrityError(
            "arbitrary exact fractions cannot establish derived authority without semantic operands"
        )

    @classmethod
    def from_untrusted_derived(cls, display: Decimal) -> "DecisionValue":
        try:
            display = _canonicalized_decision_decimal(
                validate_bounded_decimal(display, "untrusted derived display")
            )
        except (TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError(str(exception)) from exception
        payload = _decision_value_payload(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.UNTRUSTED_DERIVED,
            semantic_kind=None,
            operands=(),
            display=display,
            exact_numerator=None,
            exact_denominator=None,
        )
        return cls(
            schema_version=DECISION_VALUE_SCHEMA_VERSION,
            certainty=DecisionCertainty.UNTRUSTED_DERIVED,
            semantic_kind=None,
            operands=(),
            display=display,
            exact_numerator=None,
            exact_denominator=None,
            integrity_hash=_decision_value_hash(payload),
        )

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> "DecisionValue":
        """Restore the strict version-3 JSON-safe semantic schema."""

        if not isinstance(fields, Mapping) or set(fields) != _DECISION_VALUE_FIELDS:
            raise DecisionValueIntegrityError("decision value fields do not match schema version 3")
        try:
            certainty = DecisionCertainty(fields["certainty"])
        except (TypeError, ValueError) as exception:
            raise DecisionValueIntegrityError("decision certainty is invalid") from exception
        raw_kind = fields["semantic_kind"]
        if raw_kind is None:
            semantic_kind = None
        else:
            try:
                semantic_kind = DecisionSemanticKind(raw_kind)
            except (TypeError, ValueError) as exception:
                raise DecisionValueIntegrityError("derived decision semantic kind is invalid") from exception
        raw_operands = fields["operands"]
        expected_operand_count = 0 if semantic_kind is None else len(_operand_spec(semantic_kind))
        if (
            not isinstance(raw_operands, (list, tuple))
            or len(raw_operands) != expected_operand_count
        ):
            raise DecisionValueIntegrityError(
                "derived decision operands do not match semantic kind"
            )
        operands: list[tuple[str, str]] = []
        for pair in raw_operands:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str)
            ):
                raise DecisionValueIntegrityError("derived decision operands must be string pairs")
            operands.append((pair[0], pair[1]))
        try:
            return cls(
                schema_version=fields["schema_version"],
                certainty=certainty,
                semantic_kind=semantic_kind,
                operands=tuple(operands),
                display=_parse_canonical_decimal(fields["display"], "decision display"),
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
        self.validate_integrity()
        fields = self._payload()
        fields["integrity_hash"] = self.integrity_hash
        return fields

    def verified_operand_values(
        self,
        semantic_kind: DecisionSemanticKind,
    ) -> dict[str, Decimal]:
        """Return a fresh, revalidated operand mapping for one exact use-site role."""

        self.validate_integrity()
        if (
            self.certainty is not DecisionCertainty.EXACT_DERIVED
            or self.semantic_kind is not semantic_kind
        ):
            raise DecisionValueIntegrityError(
                "decision value does not have exact authority for the requested semantic role"
            )
        return _operand_values(semantic_kind, self.operands)

    def _operand_display(self, other: object) -> Decimal | None:
        if isinstance(other, DecisionValue):
            other.validate_integrity()
            return other.display
        if isinstance(other, Decimal):
            try:
                return validate_bounded_decimal(other, "decision arithmetic operand")
            except (TypeError, ValueError) as exception:
                raise DecisionValueIntegrityError(str(exception)) from exception
        return None

    def _binary(
        self,
        other: object,
        operation: str,
        *,
        reflected: bool = False,
    ) -> "DecisionValue":
        other_display = self._operand_display(other)
        if other_display is None:
            return NotImplemented
        if isinstance(other, Decimal) and other.is_zero():
            if operation == "add" or (operation == "subtract" and not reflected):
                return self
        left, right = (
            (other_display, self.display) if reflected else (self.display, other_display)
        )
        with decision_decimal_context():
            display = left + right if operation == "add" else left - right
        if display.is_zero():
            display = display.copy_abs()
        return DecisionValue.from_untrusted_derived(display)

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
        if exact == 0:
            return self
        with decision_decimal_context():
            display = -self.display
        if display.is_zero():
            display = display.copy_abs()
        return DecisionValue.from_untrusted_derived(display)

    def __abs__(self) -> "DecisionValue":
        exact = self.exact_fraction
        if exact is not None and exact >= 0:
            return self
        with decision_decimal_context():
            display = abs(self.display)
        return DecisionValue.from_untrusted_derived(display)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DecisionValue):
            return (
                self.certainty is other.certainty
                and self.semantic_kind is other.semantic_kind
                and self.operands == other.operands
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
            try:
                return Fraction(validate_bounded_decimal(other, "decision comparison operand"))
            except (TypeError, ValueError) as exception:
                raise DecisionValueIntegrityError(str(exception)) from exception
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
