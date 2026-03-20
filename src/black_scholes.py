"""
Black-Scholes option pricing and related functions.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European call option price.

    Parameters
    ----------
    S : float - Total equity value (underlying asset price)
    K : float - Strike price (breakpoint value)
    T : float - Time to expiry in years
    r : float - Risk-free interest rate (annualized)
    sigma : float - Volatility (annualized)

    Returns
    -------
    float - Call option price
    """
    if S <= 0:
        return 0.0
    if K <= 0:
        return S
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call_price, 0.0)


def call_spread_value(S: float, K_low: float, K_high: float,
                      T: float, r: float, sigma: float) -> float:
    """
    Value of a call spread (long call at K_low, short call at K_high).
    Represents the value of a tranche between two breakpoints.

    Parameters
    ----------
    S : float - Total equity value
    K_low : float - Lower strike (breakpoint)
    K_high : float - Upper strike (breakpoint)
    T, r, sigma : float - Black-Scholes parameters

    Returns
    -------
    float - Call spread value (always >= 0)
    """
    if K_high <= K_low:
        return 0.0
    val = black_scholes_call(S, K_low, T, r, sigma) - black_scholes_call(S, K_high, T, r, sigma)
    return max(val, 0.0)


def finnerty_dlom(sigma: float, T: float) -> float:
    """
    Finnerty (2012) average-strike put option model for DLOM estimation.

    Parameters
    ----------
    sigma : float - Volatility (annualized)
    T : float - Restriction period in years

    Returns
    -------
    float - Discount for lack of marketability as a decimal (e.g., 0.25 = 25%)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    v = sigma * np.sqrt(T)
    # Finnerty (2012) simplified protective put approximation
    dlom = 2.0 * norm.cdf(v / 2.0) - 1.0
    return max(min(dlom, 0.50), 0.0)  # Cap at 50%
