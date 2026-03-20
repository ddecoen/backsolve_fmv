"""
Data models for the Backsolve FMV Calculator.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class ParticipationType(Enum):
    NON_PARTICIPATING = "non_participating"
    FULL_PARTICIPATING = "full_participating"
    CAPPED_PARTICIPATING = "capped_participating"


@dataclass
class EquityClass:
    """Represents a class of equity (common or preferred stock)."""
    name: str
    shares_outstanding: int
    is_preferred: bool = False
    liquidation_preference_per_share: float = 0.0
    liquidation_multiple: float = 1.0
    participation: ParticipationType = ParticipationType.NON_PARTICIPATING
    participation_cap_multiple: float = 0.0  # Only used for CAPPED_PARTICIPATING
    seniority: int = 0  # Higher = more senior in liquidation waterfall
    conversion_ratio: float = 1.0  # Preferred-to-common conversion ratio

    @property
    def total_liquidation_preference(self) -> float:
        """Total aggregate liquidation preference for this class."""
        return self.shares_outstanding * self.liquidation_preference_per_share * self.liquidation_multiple

    @property
    def as_converted_shares(self) -> float:
        """Number of common shares if this class converts."""
        return self.shares_outstanding * self.conversion_ratio

    @property
    def total_participation_cap(self) -> float:
        """Total participation cap amount (only for capped participating)."""
        if self.participation == ParticipationType.CAPPED_PARTICIPATING and self.participation_cap_multiple > 0:
            return self.shares_outstanding * self.liquidation_preference_per_share * self.participation_cap_multiple
        return float('inf')


@dataclass
class OptionGrant:
    """Represents an option pool or warrant grant."""
    name: str
    shares_outstanding: int  # Granted/exercisable shares
    exercise_price: float
    shares_available: int = 0  # Ungranted shares in pool


@dataclass
class CapTable:
    """Complete capitalization table."""
    equity_classes: List[EquityClass] = field(default_factory=list)
    options: List[OptionGrant] = field(default_factory=list)
    warrants: List[OptionGrant] = field(default_factory=list)

    @property
    def common_classes(self) -> List[EquityClass]:
        return [ec for ec in self.equity_classes if not ec.is_preferred]

    @property
    def preferred_classes(self) -> List[EquityClass]:
        return [ec for ec in self.equity_classes if ec.is_preferred]

    @property
    def preferred_by_seniority(self) -> List[EquityClass]:
        """Preferred classes sorted by seniority, most senior first."""
        return sorted(self.preferred_classes, key=lambda x: x.seniority, reverse=True)

    @property
    def total_common_shares(self) -> int:
        return sum(ec.shares_outstanding for ec in self.equity_classes if not ec.is_preferred)

    @property
    def total_preferred_shares(self) -> int:
        return sum(ec.shares_outstanding for ec in self.equity_classes if ec.is_preferred)

    @property
    def total_as_converted_shares(self) -> float:
        """Total shares on a fully-diluted, as-converted basis."""
        total = sum(ec.as_converted_shares for ec in self.equity_classes)
        total += sum(op.shares_outstanding for op in self.options)
        total += sum(w.shares_outstanding for w in self.warrants)
        return total

    @property
    def total_liquidation_preference(self) -> float:
        return sum(ec.total_liquidation_preference for ec in self.preferred_classes)

    def get_class(self, name: str) -> Optional[EquityClass]:
        for ec in self.equity_classes:
            if ec.name == name:
                return ec
        return None

    def get_option(self, name: str) -> Optional[OptionGrant]:
        for op in self.options:
            if op.name == name:
                return op
        return None


@dataclass
class ValuationParams:
    """Parameters for the backsolve valuation."""
    volatility: float          # Annual volatility (e.g., 0.50 for 50%)
    risk_free_rate: float      # Annual risk-free rate (e.g., 0.045 for 4.5%)
    time_to_liquidity: float   # Years to expected liquidity event
    known_share_price: float   # Known price per share from recent transaction
    known_class_name: str      # Which class the known price applies to
    dlom_percent: float = 0.0  # Discount for lack of marketability (e.g., 0.25 for 25%)
    # Secondary transaction (hybrid weighted approach)
    secondary_price: float = 0.0        # Secondary common stock transaction price per share
    secondary_weight: float = 0.0       # Weight for secondary (0.0 to 1.0), OPM weight = 1 - this


@dataclass
class ValuationResult:
    """Output of the backsolve valuation."""
    total_equity_value: float
    per_share_values: Dict[str, float]     # class_name -> price per share (pre-DLOM)
    common_fmv: float                       # Common FMV per share after DLOM
    breakpoints: List[float]
    tranche_values: List[float]             # Value of each tranche
    allocations: List[Dict[str, float]]     # Per-tranche allocation fractions
    class_total_values: Dict[str, float]    # class_name -> total value allocated
    # Hybrid weighted approach fields
    opm_indicated_common: float = 0.0       # OPM-indicated common FMV (pre-DLOM)
    secondary_price_used: float = 0.0       # Secondary transaction price
    secondary_weight: float = 0.0           # Weight given to secondary
    blended_common_pre_dlom: float = 0.0    # Blended value before DLOM
