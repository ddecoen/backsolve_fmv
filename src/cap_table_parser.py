"""
Parser for Carta CSV exports and manual cap table input.

Supports:
  - Parsing Carta-style CSV exports with equity classes, options, and warrants
  - Building cap tables programmatically from structured dictionaries
  - Validating cap table completeness and consistency
  - Generating sample cap tables for demo/testing purposes
"""

from __future__ import annotations

import io
import math
from typing import Any, Union

import pandas as pd

from src.models import CapTable, EquityClass, OptionGrant, ParticipationType


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_liq_pref_multiple(raw: Any) -> float:
    """Convert a liquidation-preference string like '1x' or '1.5x' to float."""
    if raw is None:
        return 1.0
    if isinstance(raw, (int, float)):
        if math.isnan(raw):
            return 1.0
        return float(raw)
    raw_str = str(raw).strip().lower()
    if raw_str in ("", "nan", "none"):
        return 1.0
    if raw_str.endswith("x"):
        raw_str = raw_str[:-1]
    try:
        return float(raw_str)
    except ValueError:
        return 1.0


def _parse_participation(raw: Any) -> ParticipationType:
    """Map a free-text participation string to the ParticipationType enum."""
    if raw is None:
        return ParticipationType.NON_PARTICIPATING
    text = str(raw).strip().lower()
    if text in ("", "nan", "none"):
        return ParticipationType.NON_PARTICIPATING
    if "full" in text:
        return ParticipationType.FULL_PARTICIPATING
    if "cap" in text:
        return ParticipationType.CAPPED_PARTICIPATING
    return ParticipationType.NON_PARTICIPATING


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely coerce val to float, returning default on failure/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely coerce val to int, returning default on failure/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else int(f)
    except (ValueError, TypeError):
        return default


def _ensure_col(df: pd.DataFrame, col: str, default: Any) -> None:
    """Add col to df with default fill if it does not already exist."""
    if col not in df.columns:
        df[col] = default


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_carta_csv(uploaded_file: Union[io.IOBase, Any]) -> CapTable:
    """Parse a Carta-style CSV export into a CapTable.

    Rows whose Type column is "Option" or "Warrant" (case-insensitive) are
    parsed as option/warrant grants. All other rows are treated as equity classes.

    Missing columns are silently filled with sensible defaults.
    """
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    _ensure_col(df, "Share Class", "")
    _ensure_col(df, "Type", "Common")
    _ensure_col(df, "Shares Outstanding", 0)
    _ensure_col(df, "Issue Price", float("nan"))
    _ensure_col(df, "Liquidation Preference Multiple", float("nan"))
    _ensure_col(df, "Seniority", 0)
    _ensure_col(df, "Conversion Ratio", 1.0)
    _ensure_col(df, "Participation", "")
    _ensure_col(df, "Participation Cap", float("nan"))
    _ensure_col(df, "Shares Available", 0)

    # Split into equity rows vs option/warrant rows
    type_lower = df["Type"].astype(str).str.strip().str.lower()
    is_option_or_warrant = type_lower.isin(("option", "warrant"))

    equity_df = df[~is_option_or_warrant].copy()
    grant_df = df[is_option_or_warrant].copy()

    # Parse equity classes
    equity_classes: list[EquityClass] = []
    for _, row in equity_df.iterrows():
        name = str(row["Share Class"]).strip()
        class_type = str(row["Type"]).strip().lower()
        is_preferred = class_type == "preferred"

        issue_price = _safe_float(row["Issue Price"], 0.0)
        liq_multiple = _parse_liq_pref_multiple(row["Liquidation Preference Multiple"])
        seniority = _safe_int(row["Seniority"], 0)
        conversion_ratio = _safe_float(row["Conversion Ratio"], 1.0)
        participation = _parse_participation(row["Participation"])
        participation_cap = _safe_float(row["Participation Cap"], 0.0)

        equity_classes.append(
            EquityClass(
                name=name,
                shares_outstanding=_safe_int(row["Shares Outstanding"], 0),
                is_preferred=is_preferred,
                liquidation_preference_per_share=issue_price if is_preferred else 0.0,
                liquidation_multiple=liq_multiple,
                participation=participation,
                participation_cap_multiple=participation_cap,
                seniority=seniority,
                conversion_ratio=conversion_ratio,
            )
        )

    # Parse option / warrant grants
    options: list[OptionGrant] = []
    warrants: list[OptionGrant] = []

    for _, row in grant_df.iterrows():
        grant = OptionGrant(
            name=str(row["Share Class"]).strip(),
            shares_outstanding=_safe_int(row["Shares Outstanding"], 0),
            exercise_price=_safe_float(row["Issue Price"], 0.0),
            shares_available=_safe_int(row["Shares Available"], 0),
        )
        row_type = str(row["Type"]).strip().lower()
        if row_type == "warrant":
            warrants.append(grant)
        else:
            options.append(grant)

    return CapTable(
        equity_classes=equity_classes,
        options=options,
        warrants=warrants,
    )


# ---------------------------------------------------------------------------
# Manual / Streamlit form builder
# ---------------------------------------------------------------------------

def build_cap_table_from_inputs(
    equity_data: list[dict],
    option_data: list[dict] | None = None,
    warrant_data: list[dict] | None = None,
) -> CapTable:
    """Build a CapTable from lists of plain dictionaries.

    This is the entry-point used by the Streamlit manual-input form.
    """
    option_data = option_data or []
    warrant_data = warrant_data or []

    equity_classes: list[EquityClass] = []
    for d in equity_data:
        class_type = str(d.get("Type", d.get("type", "Common"))).strip().lower()
        is_preferred = class_type == "preferred"
        issue_price = _safe_float(
            d.get("Issue Price", d.get("issue_price",
                d.get("liquidation_preference_per_share", 0.0)))
        )
        liq_multiple = _parse_liq_pref_multiple(
            d.get("Liquidation Preference Multiple",
                d.get("liquidation_multiple", 1.0))
        )
        participation_cap = _safe_float(
            d.get("Participation Cap", d.get("participation_cap_multiple", 0.0))
        )

        equity_classes.append(
            EquityClass(
                name=str(d.get("Share Class", d.get("name", "Unknown"))).strip(),
                shares_outstanding=_safe_int(
                    d.get("Shares Outstanding", d.get("shares_outstanding", 0))
                ),
                is_preferred=is_preferred,
                liquidation_preference_per_share=issue_price if is_preferred else 0.0,
                liquidation_multiple=liq_multiple,
                participation=_parse_participation(
                    d.get("Participation", d.get("participation"))
                ),
                participation_cap_multiple=participation_cap,
                seniority=_safe_int(d.get("Seniority", d.get("seniority", 0))),
                conversion_ratio=_safe_float(
                    d.get("Conversion Ratio", d.get("conversion_ratio", 1.0)), 1.0
                ),
            )
        )

    options: list[OptionGrant] = []
    for d in option_data:
        options.append(
            OptionGrant(
                name=str(d.get("Share Class", d.get("name", "Option Pool"))).strip(),
                shares_outstanding=_safe_int(
                    d.get("Shares Outstanding", d.get("shares_outstanding", 0))
                ),
                exercise_price=_safe_float(
                    d.get("Issue Price", d.get("exercise_price", 0.0))
                ),
                shares_available=_safe_int(
                    d.get("Shares Available", d.get("shares_available", 0))
                ),
            )
        )

    wrnts: list[OptionGrant] = []
    for d in warrant_data:
        wrnts.append(
            OptionGrant(
                name=str(d.get("Share Class", d.get("name", "Warrant"))).strip(),
                shares_outstanding=_safe_int(
                    d.get("Shares Outstanding", d.get("shares_outstanding", 0))
                ),
                exercise_price=_safe_float(
                    d.get("Issue Price", d.get("exercise_price", 0.0))
                ),
                shares_available=_safe_int(
                    d.get("Shares Available", d.get("shares_available", 0))
                ),
            )
        )

    return CapTable(
        equity_classes=equity_classes,
        options=options,
        warrants=wrnts,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cap_table(cap_table: CapTable) -> list[str]:
    """Return a list of warning / error messages for the cap table.

    An empty list means the cap table passed all checks.
    """
    errors: list[str] = []

    if not cap_table.equity_classes:
        errors.append("Cap table must contain at least one equity class.")
        return errors

    common_classes = [ec for ec in cap_table.equity_classes if not ec.is_preferred]
    if not common_classes:
        errors.append("Cap table must contain at least one Common equity class.")

    seen_names: set[str] = set()
    for ec in cap_table.equity_classes:
        prefix = f"Equity class '{ec.name}'"

        if ec.name in seen_names:
            errors.append(f"{prefix}: duplicate class name.")
        seen_names.add(ec.name)

        if ec.shares_outstanding <= 0:
            errors.append(f"{prefix}: shares_outstanding must be > 0 (got {ec.shares_outstanding}).")

        if ec.is_preferred:
            if ec.liquidation_preference_per_share <= 0:
                errors.append(
                    f"{prefix}: preferred class must have a positive issue price "
                    f"(got {ec.liquidation_preference_per_share})."
                )
            if ec.seniority < 1:
                errors.append(
                    f"{prefix}: preferred class should have seniority >= 1 (got {ec.seniority})."
                )

        if ec.conversion_ratio <= 0:
            errors.append(f"{prefix}: conversion_ratio must be > 0 (got {ec.conversion_ratio}).")

        if ec.participation == ParticipationType.CAPPED_PARTICIPATING:
            if ec.participation_cap_multiple <= 0:
                errors.append(
                    f"{prefix}: capped participating class must have a positive participation_cap_multiple."
                )

    for grant in cap_table.options + cap_table.warrants:
        prefix = f"Grant '{grant.name}'"
        if grant.shares_outstanding <= 0:
            errors.append(f"{prefix}: shares_outstanding must be > 0 (got {grant.shares_outstanding}).")
        if grant.exercise_price < 0:
            errors.append(f"{prefix}: exercise_price must be >= 0 (got {grant.exercise_price}).")

    return errors


# ---------------------------------------------------------------------------
# Sample / demo data
# ---------------------------------------------------------------------------

def get_sample_cap_table() -> CapTable:
    """Return a hardcoded sample cap table suitable for demos and tests.

    Structure:
    - Common Stock: 8,000,000 shares
    - Series Seed: 1,500,000 shares, $0.50 LP/share, 1x, seniority 1
    - Series A: 3,000,000 shares, $2.00 LP/share, 1x, seniority 2
    - Series B: 2,000,000 shares, $5.00 LP/share, 1x, seniority 3
    - Option Pool: 1,200,000 shares, $0.75 exercise price
    - Warrants: 300,000 shares, $1.50 exercise price
    """
    equity_classes = [
        EquityClass(
            name="Common Stock",
            shares_outstanding=8_000_000,
            is_preferred=False,
        ),
        EquityClass(
            name="Series Seed",
            shares_outstanding=1_500_000,
            is_preferred=True,
            liquidation_preference_per_share=0.50,
            liquidation_multiple=1.0,
            seniority=1,
            conversion_ratio=1.0,
            participation=ParticipationType.NON_PARTICIPATING,
        ),
        EquityClass(
            name="Series A",
            shares_outstanding=3_000_000,
            is_preferred=True,
            liquidation_preference_per_share=2.00,
            liquidation_multiple=1.0,
            seniority=2,
            conversion_ratio=1.0,
            participation=ParticipationType.NON_PARTICIPATING,
        ),
        EquityClass(
            name="Series B",
            shares_outstanding=2_000_000,
            is_preferred=True,
            liquidation_preference_per_share=5.00,
            liquidation_multiple=1.0,
            seniority=3,
            conversion_ratio=1.0,
            participation=ParticipationType.NON_PARTICIPATING,
        ),
    ]

    options = [
        OptionGrant(
            name="2023 Option Pool",
            shares_outstanding=1_200_000,
            exercise_price=0.75,
            shares_available=500_000,
        ),
    ]

    warrants = [
        OptionGrant(
            name="Warrants",
            shares_outstanding=300_000,
            exercise_price=1.50,
            shares_available=0,
        ),
    ]

    return CapTable(
        equity_classes=equity_classes,
        options=options,
        warrants=warrants,
    )
