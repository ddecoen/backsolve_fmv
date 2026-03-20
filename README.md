# 📊 Backsolve FMV Calculator

A professional-grade **Option Pricing Method (OPM) Backsolve** tool for determining the Fair Market Value (FMV) of common stock. Built with Streamlit for an interactive, analyst-friendly experience.

---

## What Is a Backsolve FMV?

When a private company raises a new round of funding, the price paid by investors for preferred stock is known — but the fair market value of **common stock** is not directly observable. The **OPM Backsolve** technique works backwards from the known transaction price to infer the total equity value of the company, then allocates that value across all share classes using an option-pricing framework.

This is the most widely accepted methodology (per the AICPA *Valuation of Privately-Held-Company Equity Securities Issued as Compensation* guide) for 409A valuations when a recent financing round is available.

---

## Features

| Feature | Description |
|---|---|
| **Cap Table Input** | Manual entry or Carta CSV upload |
| **OPM Backsolve Engine** | Black-Scholes-based equity allocation across breakpoints |
| **DLOM Calculation** | Manual input or automated Finnerty put-option model |
| **Sensitivity Analysis** | Heatmap of FMV across volatility × time-to-liquidity grid |
| **Waterfall Visualization** | Interactive Plotly charts showing value flow through the capital structure |
| **Breakpoint Analysis** | Detailed tranche-by-tranche allocation table |
| **Sample Data** | One-click sample cap table for quick exploration |

---

## Installation

```bash
# Clone the repository
git clone <repo-url> backsolve_fmv
cd backsolve_fmv

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- See `requirements.txt` for the full dependency list.

---

## Usage

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Quick Start

1. Click **📋 Load Sample Data** in the sidebar to populate a demo cap table.
2. Switch to the **Valuation Parameters** tab and configure your assumptions (or keep the defaults).
3. Click **🚀 Run Valuation** in the sidebar.
4. Explore the **Results & Analysis** tab — metrics, waterfall, pie chart, sensitivity heatmap, and breakpoint table.

---

## Methodology

### OPM Backsolve — How It Works

1. **Build the equity waterfall.** From the cap table, compute the *breakpoints* — the enterprise-value thresholds where each equity class begins to participate (e.g., after liquidation preferences are satisfied).

2. **Model each tranche as a call-option spread.** The value of equity between two adjacent breakpoints is equivalent to a call-option spread struck at those breakpoints. The Black-Scholes formula prices each spread.

3. **Allocate tranche values.** Within each tranche, value is divided among the classes that participate at that level, pro-rata by their as-converted share counts.

4. **Backsolve for total equity value.** Given the known per-share price of the transaction class, the model iterates (using a numerical root-finder) to find the total equity value that reproduces that price.

5. **Derive common-stock FMV.** Sum the value allocated to common stock, divide by common shares, and apply a **Discount for Lack of Marketability (DLOM)** to arrive at FMV.

### DLOM — Finnerty Model

The Finnerty (2012) average-strike put-option model estimates DLOM as a function of volatility and the expected holding period. It is widely used in valuation practice for its theoretical grounding and simplicity.

---

## Carta CSV Input Format

The CSV upload expects a file exported from **Carta** (or similarly structured) with at least the following columns:

| Column | Description | Example |
|---|---|---|
| `share_class` | Name of the equity class | `Series A Preferred` |
| `class_type` | `Common` or `Preferred` | `Preferred` |
| `shares_outstanding` | Number of shares | `1000000` |
| `issue_price` | Original issue price per share | `1.50` |
| `liquidation_preference` | Liq pref multiple (e.g., 1×) | `1.0` |
| `seniority` | Seniority ranking (1 = most junior) | `2` |
| `conversion_ratio` | Preferred-to-common conversion ratio | `1.0` |
| `participation` | `Non-Participating`, `Full Participation`, or `Capped Participation` | `Non-Participating` |

Option/warrant rows (if present) should include `share_class`, `shares_outstanding`, and `exercise_price`.

---

## Screenshot

> _Screenshot placeholder — run the app and capture the Results & Analysis tab._

![screenshot](docs/screenshot.png)

---

## Project Structure

```
backsolve_fmv/
├── app.py                  # Streamlit UI
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── src/
│   ├── models.py           # Data models (CapTable, EquityClass, …)
│   ├── opm.py              # OPM breakpoint & backsolve engine
│   ├── black_scholes.py    # Black-Scholes pricing & Finnerty DLOM
│   └── cap_table_parser.py # CSV parsing & cap-table construction
├── tests/
│   └── ...                 # Unit & integration tests
└── data/
    └── ...                 # Sample data files
```

---

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — Interactive web UI
- **[NumPy](https://numpy.org/)** — Numerical computation
- **[SciPy](https://scipy.org/)** — Root-finding for backsolve
- **[Pandas](https://pandas.pydata.org/)** — Data wrangling
- **[Plotly](https://plotly.com/python/)** — Interactive charts
- **[pytest](https://pytest.org/)** — Testing

---

## Limitations & Disclaimers

> **⚠️ This tool is for educational and illustrative purposes only.**

- This is a **simplified implementation** of the OPM backsolve method. Production 409A valuations require additional considerations (e.g., probability-weighted scenarios, secondary-transaction data, additional DLOM calibration).
- The model assumes a **single liquidity event** and uses the standard Black-Scholes framework (log-normal returns, constant volatility).
- **Not financial, tax, or legal advice.** Always engage a qualified independent valuation firm for 409A or financial-reporting purposes.
- Cap structures with complex features (participating preferred with caps, pay-to-play provisions, anti-dilution ratchets) may not be fully captured.
- The Finnerty DLOM model is one of several accepted approaches; results may differ from other DLOM methodologies.

---

## License

MIT

---

*Built with ❤️ for finance and valuation professionals.*
