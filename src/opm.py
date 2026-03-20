"""
Option Pricing Method (OPM) Backsolve Engine.

Implements the OPM backsolve approach for 409A fair market value determination.
Supports non-participating, full-participating, and capped-participating preferred stock,
along with options and warrants.
"""

import numpy as np
from scipy.optimize import brentq
from typing import List, Dict, Tuple, Optional

from src.models import (
    CapTable, EquityClass, OptionGrant, ValuationParams,
    ValuationResult, ParticipationType,
)
from src.black_scholes import black_scholes_call, call_spread_value


def compute_breakpoints(cap_table: CapTable) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Compute OPM breakpoints and per-tranche allocation fractions.

    The breakpoints represent total equity values at which the marginal dollar
    distribution changes. Between consecutive breakpoints, each equity class
    receives a fixed fraction of the incremental value.

    Returns
    -------
    breakpoints : list of float
        Sorted equity value thresholds (starts at 0).
    allocations : list of dict
        allocations[i] maps class_name -> fraction of marginal dollar
        in the tranche between breakpoints[i] and breakpoints[i+1].
        The last entry covers the tranche above the final breakpoint.
    """
    all_names = _all_class_names(cap_table)
    preferred_sorted = cap_table.preferred_by_seniority  # most senior first

    # ── Phase 0: edge case — no preferred ────────────────────────────────
    if not preferred_sorted:
        bp = [0.0]
        alloc = _pro_rata_allocation_all_converted(cap_table, set())
        return bp, [alloc]

    # ── Phase 1: liquidation-preference tranches (senior → junior) ───────
    breakpoints: List[float] = [0.0]
    allocations: List[Dict[str, float]] = []
    cumulative_lp = 0.0

    for pref in preferred_sorted:
        lp = pref.total_liquidation_preference
        if lp <= 0:
            continue
        alloc = {n: 0.0 for n in all_names}
        alloc[pref.name] = 1.0
        allocations.append(alloc)
        cumulative_lp += lp
        breakpoints.append(cumulative_lp)

    total_lp = cumulative_lp

    # ── Phase 2: above total LP — depends on participation type ──────────
    #
    # For NON-PARTICIPATING preferred:
    #   Above total LP only common (+ participating preferred) shares.
    #   Each non-participating preferred converts at its conversion point.
    #
    # For PARTICIPATING preferred:
    #   Preferred participates with common immediately above LP.
    #   For CAPPED, add a breakpoint at the cap.

    # Identify which classes participate above LP (before any conversions)
    non_participating = [p for p in preferred_sorted
                         if p.participation == ParticipationType.NON_PARTICIPATING]
    full_participating = [p for p in preferred_sorted
                          if p.participation == ParticipationType.FULL_PARTICIPATING]
    capped_participating = [p for p in preferred_sorted
                            if p.participation == ParticipationType.CAPPED_PARTICIPATING]

    # Total as-converted shares assuming ALL classes convert
    total_ac = cap_table.total_as_converted_shares
    if total_ac <= 0:
        total_ac = 1.0  # prevent division by zero

    # ── Compute conversion points for non-participating preferred ────────
    # Conversion point: equity value where class is indifferent between
    # taking LP and converting to common.
    # V_convert = LP / (as_converted_shares / total_as_converted_shares)
    conversion_events: List[Tuple[float, EquityClass]] = []
    for pref in non_participating:
        if pref.total_liquidation_preference <= 0:
            continue
        ownership_frac = pref.as_converted_shares / total_ac
        if ownership_frac <= 0:
            continue
        conversion_value = pref.total_liquidation_preference / ownership_frac
        conversion_events.append((conversion_value, pref))

    conversion_events.sort(key=lambda x: x[0])

    # ── Compute participation-cap breakpoints ────────────────────────────
    cap_events: List[Tuple[float, EquityClass]] = []
    for pref in capped_participating:
        cap_amount = pref.total_participation_cap
        if cap_amount < float('inf'):
            # Breakpoint = total equity value at which the participating returns
            # (LP + share of upside) hit the cap.
            # Approximate: cap_equity_value = total_lp + (cap_amount - LP) / participation_fraction
            lp = pref.total_liquidation_preference
            # Participation fraction of this class above LP
            participating_shares = (
                sum(c.shares_outstanding for c in cap_table.common_classes) +
                sum(p.as_converted_shares for p in full_participating) +
                sum(p.as_converted_shares for p in capped_participating) +
                sum(o.shares_outstanding for o in cap_table.options) +
                sum(w.shares_outstanding for w in cap_table.warrants)
            )
            if participating_shares > 0:
                pref_frac = pref.as_converted_shares / participating_shares
                if pref_frac > 0:
                    cap_equity = total_lp + (cap_amount - lp) / pref_frac
                    cap_events.append((cap_equity, pref))

    cap_events.sort(key=lambda x: x[0])

    # ── Build tranches above total LP ────────────────────────────────────
    # Track which classes have converted and which have hit their cap
    converted_classes: set = set()
    capped_classes: set = set()

    # Merge conversion_events and cap_events into a single sorted timeline
    all_events: List[Tuple[float, str, EquityClass]] = []
    for val, ec in conversion_events:
        if val > total_lp:
            all_events.append((val, 'convert', ec))
    for val, ec in cap_events:
        if val > total_lp:
            all_events.append((val, 'cap', ec))
    all_events.sort(key=lambda x: x[0])

    # Deduplicate breakpoints that are very close
    prev_bp = total_lp
    for event_val, event_type, event_class in all_events:
        if event_val - prev_bp < 1e-6:
            # Apply the event but don't add a new breakpoint
            if event_type == 'convert':
                converted_classes.add(event_class.name)
            elif event_type == 'cap':
                capped_classes.add(event_class.name)
            continue

        # Add allocation for tranche [prev_bp, event_val]
        alloc = _compute_tranche_allocation(
            cap_table, converted_classes, capped_classes,
            full_participating, capped_participating, all_names
        )
        allocations.append(alloc)
        breakpoints.append(event_val)
        prev_bp = event_val

        # Apply the event
        if event_type == 'convert':
            converted_classes.add(event_class.name)
        elif event_type == 'cap':
            capped_classes.add(event_class.name)

    # ── Final tranche: above all events (everyone pro rata on as-converted) ──
    # If all non-participating have converted, final tranche is fully pro rata
    alloc = _compute_tranche_allocation(
        cap_table, converted_classes, capped_classes,
        full_participating, capped_participating, all_names
    )
    allocations.append(alloc)

    # If no events above total_lp were added, we still need the tranche above total_lp
    # (this is already handled since we always append the final allocation)

    return breakpoints, allocations


def _all_class_names(cap_table: CapTable) -> List[str]:
    """Get all class names including options and warrants."""
    names = [ec.name for ec in cap_table.equity_classes]
    names += [op.name for op in cap_table.options]
    names += [w.name for w in cap_table.warrants]
    return names


def _compute_tranche_allocation(
    cap_table: CapTable,
    converted_classes: set,
    capped_classes: set,
    full_participating: List[EquityClass],
    capped_participating: List[EquityClass],
    all_names: List[str],
) -> Dict[str, float]:
    """
    Compute the allocation fractions for a tranche above total LP.

    In this region:
    - Common always participates
    - Converted preferred participates (as common)
    - Full participating preferred participates
    - Capped participating preferred participates (unless capped)
    - Non-participating preferred that hasn't converted does NOT participate
    - Options/warrants participate (simplified — treated as common equivalent)
    """
    alloc = {n: 0.0 for n in all_names}

    # Common shares
    total_participating = 0.0
    participating_map: Dict[str, float] = {}

    for ec in cap_table.common_classes:
        shares = float(ec.shares_outstanding)
        participating_map[ec.name] = shares
        total_participating += shares

    # Converted non-participating preferred → participates as converted common
    for ec in cap_table.preferred_classes:
        if ec.name in converted_classes:
            shares = ec.as_converted_shares
            participating_map[ec.name] = shares
            total_participating += shares
        elif ec.participation == ParticipationType.FULL_PARTICIPATING:
            shares = ec.as_converted_shares
            participating_map[ec.name] = shares
            total_participating += shares
        elif (ec.participation == ParticipationType.CAPPED_PARTICIPATING
              and ec.name not in capped_classes):
            shares = ec.as_converted_shares
            participating_map[ec.name] = shares
            total_participating += shares

    # Options and warrants (simplified: participate proportionally)
    for op in cap_table.options:
        shares = float(op.shares_outstanding)
        participating_map[op.name] = shares
        total_participating += shares

    for w in cap_table.warrants:
        shares = float(w.shares_outstanding)
        participating_map[w.name] = shares
        total_participating += shares

    if total_participating > 0:
        for name, shares in participating_map.items():
            alloc[name] = shares / total_participating

    return alloc


def _pro_rata_allocation_all_converted(
    cap_table: CapTable, exclude: set = None
) -> Dict[str, float]:
    """Pro-rata allocation on fully-diluted as-converted basis."""
    if exclude is None:
        exclude = set()
    all_names = _all_class_names(cap_table)
    alloc = {n: 0.0 for n in all_names}

    total = 0.0
    shares_map: Dict[str, float] = {}

    for ec in cap_table.equity_classes:
        if ec.name in exclude:
            continue
        s = ec.as_converted_shares
        shares_map[ec.name] = s
        total += s

    for op in cap_table.options:
        if op.name in exclude:
            continue
        s = float(op.shares_outstanding)
        shares_map[op.name] = s
        total += s

    for w in cap_table.warrants:
        if w.name in exclude:
            continue
        s = float(w.shares_outstanding)
        shares_map[w.name] = s
        total += s

    if total > 0:
        for name, s in shares_map.items():
            alloc[name] = s / total

    return alloc


def allocate_value(
    equity_value: float,
    cap_table: CapTable,
    params: ValuationParams,
) -> Dict[str, float]:
    """
    Allocate a given total equity value across all classes using OPM.

    Uses Black-Scholes call spreads on each breakpoint tranche to determine
    the value attributable to each equity class.

    Parameters
    ----------
    equity_value : float - Total enterprise equity value
    cap_table : CapTable
    params : ValuationParams

    Returns
    -------
    dict mapping class_name -> per-share value (pre-DLOM)
    """
    if equity_value <= 0:
        all_names = _all_class_names(cap_table)
        return {n: 0.0 for n in all_names}

    breakpoints, allocations = compute_breakpoints(cap_table)
    T = params.time_to_liquidity
    r = params.risk_free_rate
    sigma = params.volatility

    all_names = _all_class_names(cap_table)
    class_values = {n: 0.0 for n in all_names}

    # Value each tranche using call spreads
    for i in range(len(allocations)):
        K_low = breakpoints[i] if i < len(breakpoints) else breakpoints[-1]

        if i < len(breakpoints) - 1:
            # Bounded tranche
            K_high = breakpoints[i + 1]
            tranche_val = call_spread_value(equity_value, K_low, K_high, T, r, sigma)
        else:
            # Unbounded tranche (above last breakpoint)
            tranche_val = black_scholes_call(equity_value, K_low, T, r, sigma)

        alloc = allocations[i]
        for name, frac in alloc.items():
            class_values[name] += tranche_val * frac

    # Handle option/warrant exercise prices —
    # Options are worth max(per_share_value - exercise_price, 0)
    # We adjust by subtracting the PV of the exercise price from option value.
    # More precisely: option_value_per_share = class_value/shares - exercise_price (floored at 0)
    # But in OPM the option shares are already allocated value, so we need to
    # subtract the aggregate exercise-price component.
    per_share = {}
    for ec in cap_table.equity_classes:
        if ec.shares_outstanding > 0:
            per_share[ec.name] = class_values[ec.name] / ec.shares_outstanding
        else:
            per_share[ec.name] = 0.0

    for op in cap_table.options:
        if op.shares_outstanding > 0:
            raw_per_share = class_values[op.name] / op.shares_outstanding
            per_share[op.name] = max(raw_per_share - op.exercise_price, 0.0)
        else:
            per_share[op.name] = 0.0

    for w in cap_table.warrants:
        if w.shares_outstanding > 0:
            raw_per_share = class_values[w.name] / w.shares_outstanding
            per_share[w.name] = max(raw_per_share - w.exercise_price, 0.0)
        else:
            per_share[w.name] = 0.0

    return per_share


def backsolve_equity_value(
    cap_table: CapTable,
    params: ValuationParams,
) -> ValuationResult:
    """
    Backsolve for total equity value given a known share price.

    Uses Brent's method to find the total equity value such that the
    OPM allocation produces the known per-share price for the specified class.

    Parameters
    ----------
    cap_table : CapTable
    params : ValuationParams

    Returns
    -------
    ValuationResult
    """
    known_class = params.known_class_name
    known_price = params.known_share_price

    # Validate the known class exists
    found_class = cap_table.get_class(known_class)
    found_option = cap_table.get_option(known_class) if found_class is None else None
    if found_class is None and found_option is None:
        # Check warrants
        found_warrant = None
        for w in cap_table.warrants:
            if w.name == known_class:
                found_warrant = w
                break
        if found_warrant is None:
            raise ValueError(f"Known class '{known_class}' not found in cap table.")

    # Objective function: difference between computed and known price
    def objective(equity_value: float) -> float:
        per_share = allocate_value(equity_value, cap_table, params)
        computed_price = per_share.get(known_class, 0.0)
        return computed_price - known_price

    # Determine search bounds
    total_ac = cap_table.total_as_converted_shares
    if total_ac <= 0:
        total_ac = 1.0

    # Lower bound: at minimum, equity value should be near the LP total
    lower = max(known_price * 0.01, 1.0)
    # Upper bound: generous multiple
    upper = known_price * total_ac * 20.0

    # Ensure the objective has opposite signs at the bounds
    f_lower = objective(lower)
    f_upper = objective(upper)

    # Expand bounds if needed
    attempts = 0
    while f_lower > 0 and attempts < 20:
        lower = lower / 10.0
        f_lower = objective(lower)
        attempts += 1

    attempts = 0
    while f_upper < 0 and attempts < 20:
        upper = upper * 5.0
        f_upper = objective(upper)
        attempts += 1

    if f_lower * f_upper > 0:
        raise ValueError(
            f"Cannot find bracketing interval for backsolve. "
            f"f({lower:.2f})={f_lower:.6f}, f({upper:.2f})={f_upper:.6f}. "
            f"The known price of ${known_price:.2f} for '{known_class}' may not be achievable."
        )

    # Solve
    solved_equity_value = brentq(objective, lower, upper, xtol=1e-4, rtol=1e-8, maxiter=200)

    # Compute final allocations
    per_share = allocate_value(solved_equity_value, cap_table, params)
    breakpoints, alloc_fracs = compute_breakpoints(cap_table)

    # Compute tranche values for reporting
    T = params.time_to_liquidity
    r = params.risk_free_rate
    sigma = params.volatility

    tranche_values = []
    for i in range(len(alloc_fracs)):
        K_low = breakpoints[i] if i < len(breakpoints) else breakpoints[-1]
        if i < len(breakpoints) - 1:
            K_high = breakpoints[i + 1]
            tv = call_spread_value(solved_equity_value, K_low, K_high, T, r, sigma)
        else:
            tv = black_scholes_call(solved_equity_value, K_low, T, r, sigma)
        tranche_values.append(tv)

    # Compute total values per class
    class_total_values = {}
    for ec in cap_table.equity_classes:
        class_total_values[ec.name] = per_share.get(ec.name, 0.0) * ec.shares_outstanding
    for op in cap_table.options:
        class_total_values[op.name] = per_share.get(op.name, 0.0) * op.shares_outstanding
    for w in cap_table.warrants:
        class_total_values[w.name] = per_share.get(w.name, 0.0) * w.shares_outstanding

    # Common FMV after DLOM
    common_names = [ec.name for ec in cap_table.common_classes]
    if common_names:
        common_pps = per_share.get(common_names[0], 0.0)
    else:
        common_pps = 0.0

    dlom = params.dlom_percent
    common_fmv = common_pps * (1.0 - dlom)

    return ValuationResult(
        total_equity_value=solved_equity_value,
        per_share_values=per_share,
        common_fmv=common_fmv,
        breakpoints=breakpoints,
        tranche_values=tranche_values,
        allocations=alloc_fracs,
        class_total_values=class_total_values,
    )
