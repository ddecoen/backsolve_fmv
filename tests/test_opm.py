"""Unit tests for the backsolve FMV calculator."""

import pytest
import math
import io

from src.models import (
    CapTable, EquityClass, OptionGrant, ValuationParams,
    ValuationResult, ParticipationType,
)
from src.opm import compute_breakpoints, allocate_value, backsolve_equity_value
from src.black_scholes import black_scholes_call, call_spread_value, finnerty_dlom
from src.cap_table_parser import (
    get_sample_cap_table, validate_cap_table, parse_carta_csv,
    build_cap_table_from_inputs,
)


# ====================================================================
# Black-Scholes tests
# ====================================================================

class TestBlackScholesCall:
    def test_atm_known_value(self):
        """S=100, K=100, T=1, r=5%, sigma=20% -> ~10.45"""
        c = black_scholes_call(100, 100, 1, 0.05, 0.2)
        assert abs(c - 10.4506) < 0.01

    def test_deep_itm(self):
        c = black_scholes_call(200, 50, 1, 0.05, 0.3)
        assert c > 140  # should be close to intrinsic

    def test_deep_otm(self):
        c = black_scholes_call(10, 200, 1, 0.05, 0.3)
        assert c < 1.0

    def test_zero_underlying(self):
        assert black_scholes_call(0, 100, 1, 0.05, 0.2) == 0.0

    def test_zero_strike(self):
        assert black_scholes_call(100, 0, 1, 0.05, 0.2) == 100.0

    def test_zero_time(self):
        assert black_scholes_call(100, 90, 0, 0.05, 0.2) == 10.0
        assert black_scholes_call(100, 110, 0, 0.05, 0.2) == 0.0

    def test_higher_vol_higher_price(self):
        c_low = black_scholes_call(100, 100, 1, 0.05, 0.2)
        c_high = black_scholes_call(100, 100, 1, 0.05, 0.5)
        assert c_high > c_low

    def test_higher_time_higher_price(self):
        c_short = black_scholes_call(100, 100, 0.5, 0.05, 0.3)
        c_long = black_scholes_call(100, 100, 2.0, 0.05, 0.3)
        assert c_long > c_short


class TestCallSpreadValue:
    def test_positive_spread(self):
        v = call_spread_value(100, 50, 150, 1, 0.05, 0.3)
        assert v > 0

    def test_zero_width(self):
        v = call_spread_value(100, 80, 80, 1, 0.05, 0.3)
        assert v == 0.0

    def test_inverted_strikes(self):
        v = call_spread_value(100, 150, 50, 1, 0.05, 0.3)
        assert v == 0.0

    def test_bounded_by_components(self):
        c_low = black_scholes_call(100, 50, 1, 0.05, 0.3)
        spread = call_spread_value(100, 50, 150, 1, 0.05, 0.3)
        assert spread <= c_low


class TestFinnertyDlom:
    def test_positive_result(self):
        d = finnerty_dlom(0.5, 3.0)
        assert 0.0 < d < 0.50

    def test_zero_vol(self):
        assert finnerty_dlom(0.0, 3.0) == 0.0

    def test_zero_time(self):
        assert finnerty_dlom(0.5, 0.0) == 0.0

    def test_higher_vol_higher_dlom(self):
        d_low = finnerty_dlom(0.2, 2.0)
        d_high = finnerty_dlom(0.6, 2.0)
        assert d_high > d_low


# ====================================================================
# OPM breakpoint tests
# ====================================================================

class TestComputeBreakpoints:
    def test_simple_two_class(self):
        """Common + 1 preferred, check breakpoints."""
        ct = CapTable(equity_classes=[
            EquityClass(name="Common", shares_outstanding=8_000_000),
            EquityClass(
                name="Series A", shares_outstanding=2_000_000, is_preferred=True,
                liquidation_preference_per_share=1.0, seniority=1,
            ),
        ])
        bps, allocs = compute_breakpoints(ct)
        assert bps[0] == 0.0
        assert bps[1] == 2_000_000.0  # Total LP
        assert len(allocs) >= 2

    def test_no_preferred(self):
        ct = CapTable(equity_classes=[
            EquityClass(name="Common", shares_outstanding=10_000_000),
        ])
        bps, allocs = compute_breakpoints(ct)
        assert bps == [0.0]
        assert len(allocs) == 1
        assert allocs[0]["Common"] == 1.0

    def test_breakpoints_sorted(self):
        ct = get_sample_cap_table()
        bps, allocs = compute_breakpoints(ct)
        for i in range(1, len(bps)):
            assert bps[i] > bps[i - 1]

    def test_allocations_sum_to_one(self):
        ct = get_sample_cap_table()
        bps, allocs = compute_breakpoints(ct)
        for alloc in allocs:
            total = sum(alloc.values())
            assert abs(total - 1.0) < 1e-9


# ====================================================================
# Allocate value tests
# ====================================================================

class TestAllocateValue:
    def test_zero_equity(self):
        ct = get_sample_cap_table()
        params = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.0, known_class_name="Series B",
        )
        result = allocate_value(0.0, ct, params)
        for v in result.values():
            assert v == 0.0

    def test_all_values_non_negative(self):
        ct = get_sample_cap_table()
        params = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.0, known_class_name="Series B",
        )
        result = allocate_value(50_000_000, ct, params)
        for v in result.values():
            assert v >= 0.0

    def test_preferred_worth_more_than_common_at_low_value(self):
        ct = CapTable(equity_classes=[
            EquityClass(name="Common", shares_outstanding=8_000_000),
            EquityClass(
                name="Series A", shares_outstanding=2_000_000, is_preferred=True,
                liquidation_preference_per_share=5.0, seniority=1,
            ),
        ])
        params = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.0, known_class_name="Series A",
        )
        result = allocate_value(12_000_000, ct, params)
        # Preferred should be worth more per share than common at moderate equity value
        assert result["Series A"] > result["Common"]


# ====================================================================
# Backsolve tests
# ====================================================================

class TestBacksolveEquityValue:
    def test_backsolve_recovers_known_price(self):
        ct = get_sample_cap_table()
        params = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.00, known_class_name="Series B", dlom_percent=0.25,
        )
        result = backsolve_equity_value(ct, params)
        assert abs(result.per_share_values["Series B"] - 5.00) < 0.01

    def test_positive_equity_value(self):
        ct = get_sample_cap_table()
        params = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.00, known_class_name="Series B",
        )
        result = backsolve_equity_value(ct, params)
        assert result.total_equity_value > 0

    def test_common_fmv_less_than_preferred(self):
        ct = get_sample_cap_table()
        params = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.00, known_class_name="Series B", dlom_percent=0.25,
        )
        result = backsolve_equity_value(ct, params)
        assert result.common_fmv < result.per_share_values["Series B"]

    def test_dlom_reduces_common(self):
        ct = get_sample_cap_table()
        params_no_dlom = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.00, known_class_name="Series B", dlom_percent=0.0,
        )
        params_dlom = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.00, known_class_name="Series B", dlom_percent=0.25,
        )
        r1 = backsolve_equity_value(ct, params_no_dlom)
        r2 = backsolve_equity_value(ct, params_dlom)
        assert r2.common_fmv < r1.common_fmv

    def test_invalid_class_raises(self):
        ct = get_sample_cap_table()
        params = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.00, known_class_name="Series Z",
        )
        with pytest.raises(ValueError):
            backsolve_equity_value(ct, params)

    def test_higher_price_higher_equity(self):
        ct = get_sample_cap_table()
        p1 = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=3.00, known_class_name="Series B",
        )
        p2 = ValuationParams(
            volatility=0.5, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=8.00, known_class_name="Series B",
        )
        r1 = backsolve_equity_value(ct, p1)
        r2 = backsolve_equity_value(ct, p2)
        assert r2.total_equity_value > r1.total_equity_value


# ====================================================================
# Conversion floor tests
# ====================================================================

class TestConversionFloor:
    def test_preferred_geq_common(self):
        """All preferred per-share values must be >= common per-share."""
        ct = get_sample_cap_table()
        params = ValuationParams(
            volatility=0.7, risk_free_rate=0.045, time_to_liquidity=3.0,
            known_share_price=5.00, known_class_name="Series B", dlom_percent=0.20,
        )
        result = backsolve_equity_value(ct, params)
        common_pps = result.per_share_values["Common Stock"]
        for ec in ct.preferred_classes:
            assert result.per_share_values[ec.name] >= common_pps - 1e-6

    def test_floor_applied_when_binding(self):
        """If OPM would produce preferred < common, floor kicks in."""
        from src.opm import apply_conversion_floor
        ct = CapTable(equity_classes=[
            EquityClass(name="Common", shares_outstanding=10_000_000),
            EquityClass(
                name="Pref A", shares_outstanding=500_000, is_preferred=True,
                liquidation_preference_per_share=0.01, seniority=1,
            ),
        ])
        raw = {"Common": 2.00, "Pref A": 0.50}
        floored = apply_conversion_floor(raw, ct)
        assert floored["Pref A"] >= floored["Common"]

    def test_floor_preserves_higher_values(self):
        """Floor should not reduce preferred values that are already above common."""
        from src.opm import apply_conversion_floor
        ct = CapTable(equity_classes=[
            EquityClass(name="Common", shares_outstanding=10_000_000),
            EquityClass(
                name="Pref A", shares_outstanding=1_000_000, is_preferred=True,
                liquidation_preference_per_share=5.0, seniority=1,
            ),
        ])
        raw = {"Common": 2.00, "Pref A": 8.00}
        floored = apply_conversion_floor(raw, ct)
        assert floored["Pref A"] == 8.00  # unchanged


# ====================================================================
# Validation tests
# ====================================================================

class TestValidateCapTable:
    def test_valid_table(self):
        ct = get_sample_cap_table()
        errors = validate_cap_table(ct)
        assert errors == []

    def test_empty_table(self):
        ct = CapTable()
        errors = validate_cap_table(ct)
        assert len(errors) > 0

    def test_no_common(self):
        ct = CapTable(equity_classes=[
            EquityClass(name="Pref", shares_outstanding=1_000_000, is_preferred=True,
                        liquidation_preference_per_share=1.0, seniority=1),
        ])
        errors = validate_cap_table(ct)
        assert any("Common" in e for e in errors)

    def test_zero_shares(self):
        ct = CapTable(equity_classes=[
            EquityClass(name="Common", shares_outstanding=0),
        ])
        errors = validate_cap_table(ct)
        assert any("shares_outstanding" in e for e in errors)


# ====================================================================
# Cap table parser tests
# ====================================================================

class TestGetSampleCapTable:
    def test_returns_captable(self):
        ct = get_sample_cap_table()
        assert isinstance(ct, CapTable)

    def test_has_common(self):
        ct = get_sample_cap_table()
        assert any(not ec.is_preferred for ec in ct.equity_classes)

    def test_has_preferred(self):
        ct = get_sample_cap_table()
        assert any(ec.is_preferred for ec in ct.equity_classes)

    def test_has_options(self):
        ct = get_sample_cap_table()
        assert len(ct.options) > 0

    def test_has_warrants(self):
        ct = get_sample_cap_table()
        assert len(ct.warrants) > 0

    def test_class_count(self):
        ct = get_sample_cap_table()
        assert len(ct.equity_classes) == 4


class TestParseCartaCsv:
    def test_roundtrip(self):
        csv_data = """Share Class,Type,Shares Outstanding,Issue Price,Liquidation Preference Multiple,Seniority,Conversion Ratio,Participation,Participation Cap,Shares Available
Common Stock,Common,8000000,,,,1.0,,,
Series A,Preferred,2000000,2.00,1x,1,1.0,Non-participating,,
Option Pool,Option,500000,0.50,,,,,,100000
"""
        ct = parse_carta_csv(io.StringIO(csv_data))
        assert len(ct.equity_classes) == 2
        assert len(ct.options) == 1
        assert ct.equity_classes[0].name == "Common Stock"
        assert ct.equity_classes[1].is_preferred is True
        assert ct.equity_classes[1].liquidation_preference_per_share == 2.0
        assert ct.options[0].exercise_price == 0.50


class TestBuildCapTableFromInputs:
    def test_basic_build(self):
        equity = [
            {"Share Class": "Common", "Type": "Common", "Shares Outstanding": 1000000},
            {"Share Class": "Pref A", "Type": "Preferred", "Shares Outstanding": 500000,
             "Issue Price": 2.0, "Seniority": 1},
        ]
        ct = build_cap_table_from_inputs(equity)
        assert len(ct.equity_classes) == 2
        assert ct.equity_classes[1].is_preferred is True
