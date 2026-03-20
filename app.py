import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.models import (
    CapTable, EquityClass, OptionGrant, ValuationParams,
    ParticipationType, ValuationResult,
)
from src.opm import compute_breakpoints, allocate_value, backsolve_equity_value
from src.cap_table_parser import (
    parse_carta_csv, build_cap_table_from_inputs,
    validate_cap_table, get_sample_cap_table,
)
from src.black_scholes import finnerty_dlom

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Backsolve FMV Calculator",
    layout="wide",
    page_icon="📊",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 0.75rem; padding: 1.25rem; text-align: center;
    }
    .metric-card h3 {margin:0; color:#334155; font-size:0.85rem; font-weight:600;}
    .metric-card p  {margin:0.25rem 0 0 0; color:#0f172a; font-size:1.6rem; font-weight:700;}
    .section-hdr {
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.4rem; margin-bottom: 1rem; color: #1e40af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "cap_table": None,
    "valuation_params": None,
    "valuation_result": None,
    "equity_classes": [],
    "option_grants": [],
    "warrant_grants": [],
    "csv_uploaded": False,
    "sample_loaded": False,
    "run_requested": False,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

if not st.session_state["equity_classes"]:
    st.session_state["equity_classes"] = [
        {
            "name": "Common Stock",
            "type": "Common",
            "shares": 1_000_000,
            "issue_price": 0.0,
            "liq_pref_multiple": 1.0,
            "seniority": 0,
            "conversion_ratio": 1.0,
            "participation": "Non-participating",
        }
    ]

# ---------------------------------------------------------------------------
# Helper: participation string -> enum
# ---------------------------------------------------------------------------
_PART_OPTIONS = ["Non-participating", "Full participating", "Capped participating"]

def _part_str_to_enum(s: str) -> ParticipationType:
    s = s.lower()
    if "full" in s:
        return ParticipationType.FULL_PARTICIPATING
    if "cap" in s:
        return ParticipationType.CAPPED_PARTICIPATING
    return ParticipationType.NON_PARTICIPATING

def _enum_to_part_str(p: ParticipationType) -> str:
    return {
        ParticipationType.NON_PARTICIPATING: "Non-participating",
        ParticipationType.FULL_PARTICIPATING: "Full participating",
        ParticipationType.CAPPED_PARTICIPATING: "Capped participating",
    }.get(p, "Non-participating")


# ---------------------------------------------------------------------------
# Helper: build CapTable from session state
# ---------------------------------------------------------------------------
def _build_cap_table_from_state() -> CapTable:
    equity_classes = []
    for ec in st.session_state["equity_classes"]:
        is_pref = ec["type"] == "Preferred"
        equity_classes.append(
            EquityClass(
                name=ec["name"],
                shares_outstanding=int(ec["shares"]),
                is_preferred=is_pref,
                liquidation_preference_per_share=float(ec["issue_price"]) if is_pref else 0.0,
                liquidation_multiple=float(ec["liq_pref_multiple"]),
                participation=_part_str_to_enum(ec.get("participation", "Non-participating")),
                participation_cap_multiple=0.0,
                seniority=int(ec["seniority"]),
                conversion_ratio=float(ec["conversion_ratio"]),
            )
        )

    options = []
    for og in st.session_state["option_grants"]:
        options.append(
            OptionGrant(
                name=og["name"],
                shares_outstanding=int(og["shares"]),
                exercise_price=float(og["exercise_price"]),
            )
        )

    warrants = []
    for wg in st.session_state["warrant_grants"]:
        warrants.append(
            OptionGrant(
                name=wg["name"],
                shares_outstanding=int(wg["shares"]),
                exercise_price=float(wg["exercise_price"]),
            )
        )

    return CapTable(equity_classes=equity_classes, options=options, warrants=warrants)


# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.markdown("## 📊 Backsolve FMV Calculator")
    st.markdown(
        "Determine fair market value of common stock using the **Option Pricing "
        "Method (OPM) Backsolve** approach."
    )
    st.divider()

    # Load sample data
    if st.button("📋 Load Sample Data", use_container_width=True):
        try:
            sample = get_sample_cap_table()
            st.session_state["cap_table"] = sample
            st.session_state["equity_classes"] = [
                {
                    "name": ec.name,
                    "type": "Preferred" if ec.is_preferred else "Common",
                    "shares": ec.shares_outstanding,
                    "issue_price": ec.liquidation_preference_per_share,
                    "liq_pref_multiple": ec.liquidation_multiple,
                    "seniority": ec.seniority,
                    "conversion_ratio": ec.conversion_ratio,
                    "participation": _enum_to_part_str(ec.participation),
                }
                for ec in sample.equity_classes
            ]
            st.session_state["option_grants"] = [
                {"name": op.name, "shares": op.shares_outstanding, "exercise_price": op.exercise_price}
                for op in sample.options
            ]
            st.session_state["warrant_grants"] = [
                {"name": w.name, "shares": w.shares_outstanding, "exercise_price": w.exercise_price}
                for w in sample.warrants
            ]
            st.session_state["sample_loaded"] = True
            st.success("Sample data loaded!")
        except Exception as exc:
            st.error(f"Failed to load sample data: {exc}")

    st.divider()

    run_clicked = st.button("🚀 Run Valuation", type="primary", use_container_width=True)
    if run_clicked:
        st.session_state["run_requested"] = True


# ===================================================================
# MAIN CONTENT — three tabs
# ===================================================================
tab_cap, tab_params, tab_results = st.tabs(
    ["📋 Cap Table Input", "⚙️ Valuation Parameters", "📈 Results & Analysis"]
)

# -------------------------------------------------------------------
# TAB 1: Cap Table Input
# -------------------------------------------------------------------
with tab_cap:
    st.markdown('<h2 class="section-hdr">Cap Table Input</h2>', unsafe_allow_html=True)
    st.info(
        "Enter your company's capitalization table below. You can upload a Carta "
        "export (CSV) or fill in the details manually. The cap table drives the "
        "equity-waterfall analysis used in the OPM backsolve."
    )

    input_method = st.radio(
        "Input method", ["Manual Input", "Upload Carta CSV"],
        horizontal=True, key="input_method",
    )

    # ---- CSV upload path ------------------------------------------------
    if input_method == "Upload Carta CSV":
        uploaded = st.file_uploader("Upload your Carta cap-table CSV export", type=["csv"], key="csv_uploader")
        if uploaded is not None:
            try:
                cap_table = parse_carta_csv(uploaded)
                st.session_state["cap_table"] = cap_table
                st.session_state["csv_uploaded"] = True
                st.session_state["equity_classes"] = [
                    {
                        "name": ec.name,
                        "type": "Preferred" if ec.is_preferred else "Common",
                        "shares": ec.shares_outstanding,
                        "issue_price": ec.liquidation_preference_per_share,
                        "liq_pref_multiple": ec.liquidation_multiple,
                        "seniority": ec.seniority,
                        "conversion_ratio": ec.conversion_ratio,
                        "participation": _enum_to_part_str(ec.participation),
                    }
                    for ec in cap_table.equity_classes
                ]
                st.session_state["option_grants"] = [
                    {"name": op.name, "shares": op.shares_outstanding, "exercise_price": op.exercise_price}
                    for op in cap_table.options
                ]
                st.session_state["warrant_grants"] = [
                    {"name": w.name, "shares": w.shares_outstanding, "exercise_price": w.exercise_price}
                    for w in cap_table.warrants
                ]
                st.success("CSV parsed successfully!")
                preview_rows = []
                for ec in cap_table.equity_classes:
                    preview_rows.append({
                        "Class": ec.name,
                        "Type": "Preferred" if ec.is_preferred else "Common",
                        "Shares": f"{ec.shares_outstanding:,.0f}",
                        "LP/Share": f"${ec.liquidation_preference_per_share:,.4f}" if ec.is_preferred else "—",
                        "Seniority": ec.seniority,
                    })
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Error parsing CSV: {exc}")

    # ---- Manual input path ----------------------------------------------
    else:
        st.markdown("### Equity Classes")
        for idx, ec in enumerate(st.session_state["equity_classes"]):
            with st.container(border=True):
                cols = st.columns([2, 1.2, 1.5, 1.2, 1.2, 1, 1.2, 1.8])
                ec["name"] = cols[0].text_input("Name", value=ec["name"], key=f"ec_name_{idx}")
                ec["type"] = cols[1].selectbox(
                    "Type", ["Common", "Preferred"],
                    index=0 if ec["type"] == "Common" else 1, key=f"ec_type_{idx}",
                )
                ec["shares"] = cols[2].number_input(
                    "Shares", min_value=1, value=int(ec["shares"]), step=1000, key=f"ec_shares_{idx}",
                )
                ec["issue_price"] = cols[3].number_input(
                    "LP/Share ($)", min_value=0.0, value=float(ec["issue_price"]),
                    step=0.01, format="%.4f", key=f"ec_ip_{idx}",
                )
                ec["liq_pref_multiple"] = cols[4].number_input(
                    "LP Multiple", min_value=0.0, value=float(ec["liq_pref_multiple"]),
                    step=0.25, key=f"ec_liq_{idx}",
                )
                ec["seniority"] = cols[5].number_input(
                    "Seniority", min_value=0, value=int(ec["seniority"]), step=1, key=f"ec_sen_{idx}",
                )
                ec["conversion_ratio"] = cols[6].number_input(
                    "Conv. Ratio", min_value=0.01, value=float(ec["conversion_ratio"]),
                    step=0.1, format="%.2f", key=f"ec_cr_{idx}",
                )
                current_part = ec.get("participation", "Non-participating")
                part_index = _PART_OPTIONS.index(current_part) if current_part in _PART_OPTIONS else 0
                ec["participation"] = cols[7].selectbox(
                    "Participation", _PART_OPTIONS, index=part_index, key=f"ec_part_{idx}",
                )

        col_add_eq, col_rm_eq, _ = st.columns([1, 1, 4])
        if col_add_eq.button("➕ Add Equity Class", key="add_eq"):
            next_seniority = len([e for e in st.session_state["equity_classes"] if e["type"] == "Preferred"]) + 1
            st.session_state["equity_classes"].append({
                "name": f"Series {chr(64 + next_seniority)}",
                "type": "Preferred", "shares": 500_000, "issue_price": 1.00,
                "liq_pref_multiple": 1.0, "seniority": next_seniority,
                "conversion_ratio": 1.0, "participation": "Non-participating",
            })
            st.rerun()
        if col_rm_eq.button("➖ Remove Last Class", key="rm_eq",
                            disabled=len(st.session_state["equity_classes"]) <= 1):
            st.session_state["equity_classes"].pop()
            st.rerun()

        st.markdown("---")

        # — Options -------------------------------------------------------
        st.markdown("### Options")
        if not st.session_state["option_grants"]:
            st.caption("No option pools added yet.")
        for idx, og in enumerate(st.session_state["option_grants"]):
            with st.container(border=True):
                cols = st.columns([2, 2, 2])
                og["name"] = cols[0].text_input("Name", value=og["name"], key=f"og_name_{idx}")
                og["shares"] = cols[1].number_input(
                    "Shares Outstanding", min_value=1, value=int(og["shares"]), step=1000, key=f"og_shares_{idx}",
                )
                og["exercise_price"] = cols[2].number_input(
                    "Exercise Price ($)", min_value=0.0, value=float(og["exercise_price"]),
                    step=0.01, format="%.4f", key=f"og_ep_{idx}",
                )

        # — Warrants ------------------------------------------------------
        st.markdown("### Warrants")
        if not st.session_state["warrant_grants"]:
            st.caption("No warrants added yet.")
        for idx, wg in enumerate(st.session_state["warrant_grants"]):
            with st.container(border=True):
                cols = st.columns([2, 2, 2])
                wg["name"] = cols[0].text_input("Name", value=wg["name"], key=f"wg_name_{idx}")
                wg["shares"] = cols[1].number_input(
                    "Shares Outstanding", min_value=1, value=int(wg["shares"]), step=1000, key=f"wg_shares_{idx}",
                )
                wg["exercise_price"] = cols[2].number_input(
                    "Exercise Price ($)", min_value=0.0, value=float(wg["exercise_price"]),
                    step=0.01, format="%.4f", key=f"wg_ep_{idx}",
                )

        col_add_op, col_add_w, col_rm, _ = st.columns([1, 1, 1, 3])
        if col_add_op.button("➕ Add Option Pool", key="add_op"):
            st.session_state["option_grants"].append(
                {"name": "Employee Option Pool", "shares": 200_000, "exercise_price": 0.50}
            )
            st.rerun()
        if col_add_w.button("➕ Add Warrant", key="add_w"):
            st.session_state["warrant_grants"].append(
                {"name": "Warrant", "shares": 50_000, "exercise_price": 1.00}
            )
            st.rerun()
        if col_rm.button("➖ Remove Last Grant", key="rm_og",
                         disabled=(len(st.session_state["option_grants"]) + len(st.session_state["warrant_grants"])) == 0):
            if st.session_state["warrant_grants"]:
                st.session_state["warrant_grants"].pop()
            elif st.session_state["option_grants"]:
                st.session_state["option_grants"].pop()
            st.rerun()

    # ---- Cap-table summary (always visible) -----------------------------
    st.markdown("---")
    st.markdown("### 📊 Cap Table Summary")
    try:
        current_ct = _build_cap_table_from_state()
        st.session_state["cap_table"] = current_ct

        summary_rows = []
        total_ac = current_ct.total_as_converted_shares or 1.0

        for ec in current_ct.equity_classes:
            ac = ec.as_converted_shares
            summary_rows.append({
                "Class": ec.name,
                "Type": "Preferred" if ec.is_preferred else "Common",
                "Shares Outstanding": f"{ec.shares_outstanding:,.0f}",
                "As-Converted Shares": f"{ac:,.0f}",
                "Ownership %": f"{ac / total_ac * 100:.1f}%",
                "Total LP": f"${ec.total_liquidation_preference:,.0f}" if ec.is_preferred else "—",
            })
        for op in current_ct.options:
            summary_rows.append({
                "Class": op.name, "Type": "Option",
                "Shares Outstanding": f"{op.shares_outstanding:,.0f}",
                "As-Converted Shares": f"{op.shares_outstanding:,.0f}",
                "Ownership %": f"{op.shares_outstanding / total_ac * 100:.1f}%",
                "Total LP": "—",
            })
        for w in current_ct.warrants:
            summary_rows.append({
                "Class": w.name, "Type": "Warrant",
                "Shares Outstanding": f"{w.shares_outstanding:,.0f}",
                "As-Converted Shares": f"{w.shares_outstanding:,.0f}",
                "Ownership %": f"{w.shares_outstanding / total_ac * 100:.1f}%",
                "Total LP": "—",
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Ownership pie chart
        pie_labels = [r["Class"] for r in summary_rows]
        pie_values = [float(r["As-Converted Shares"].replace(",", "")) for r in summary_rows]
        fig_pie = px.pie(
            names=pie_labels, values=pie_values,
            title="Ownership (As-Converted Basis)",
            color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.35,
        )
        fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

        issues = validate_cap_table(current_ct)
        if issues:
            for issue in issues:
                st.warning(issue)
    except Exception as exc:
        st.error(f"Could not build cap table: {exc}")

# -------------------------------------------------------------------
# TAB 2: Valuation Parameters
# -------------------------------------------------------------------
with tab_params:
    st.markdown('<h2 class="section-hdr">Valuation Parameters</h2>', unsafe_allow_html=True)
    st.info(
        "Configure the parameters for the OPM backsolve. The model uses a known "
        "transaction price (e.g., the most recent preferred-stock round) to solve "
        "for the implied total equity value and derive common-stock FMV."
    )

    class_names = [ec["name"] for ec in st.session_state["equity_classes"]]
    if not class_names:
        class_names = ["(no classes defined)"]

    with st.form("valuation_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Transaction Details")
            known_class = st.selectbox(
                "Known Transaction Class", class_names,
                index=min(len(class_names) - 1, 1) if len(class_names) > 1 else 0,
                help="The equity class whose price-per-share is known from a recent transaction.",
            )
            known_pps = st.number_input(
                "Known Price Per Share ($)", min_value=0.01, value=5.00,
                step=0.01, format="%.4f",
                help="Price per share from the known transaction.",
            )

        with col_right:
            st.markdown("#### Market Assumptions")
            volatility = st.slider(
                "Expected Volatility (%)", min_value=20, max_value=100, value=50, step=1,
                help="Annualised equity volatility used in the OPM Black-Scholes framework.",
            )
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)", min_value=0.0, max_value=15.0,
                value=4.5, step=0.1, format="%.2f",
            )
            time_to_liquidity = st.number_input(
                "Time to Liquidity Event (years)", min_value=0.5, max_value=10.0,
                value=3.0, step=0.25, format="%.2f",
            )

        st.markdown("---")
        st.markdown("#### Discount for Lack of Marketability (DLOM)")
        dlom_method = st.radio("DLOM Method", ["Manual Input", "Finnerty Model"], horizontal=True)
        if dlom_method == "Manual Input":
            dlom_pct = st.slider("DLOM (%)", min_value=0, max_value=50, value=25, step=1,
                                 help="Manually specified DLOM to apply to common stock.")
        else:
            dlom_pct = None

        submitted = st.form_submit_button("💾 Save Parameters", use_container_width=True)

    if submitted:
        vol_dec = volatility / 100.0
        rfr_dec = risk_free_rate / 100.0
        if dlom_method == "Finnerty Model":
            try:
                dlom_value = finnerty_dlom(vol_dec, time_to_liquidity)
                st.success(f"Finnerty DLOM calculated: **{dlom_value * 100:.1f}%** (σ={vol_dec:.0%}, T={time_to_liquidity:.1f}yr)")
            except Exception as exc:
                st.error(f"Finnerty DLOM calculation failed: {exc}")
                dlom_value = 0.25
        else:
            dlom_value = dlom_pct / 100.0

        st.session_state["valuation_params"] = ValuationParams(
            volatility=vol_dec,
            risk_free_rate=rfr_dec,
            time_to_liquidity=time_to_liquidity,
            known_share_price=known_pps,
            known_class_name=known_class,
            dlom_percent=dlom_value,
        )
        st.toast("Parameters saved ✅")

    st.markdown("### 📝 Current Parameter Summary")
    vp = st.session_state.get("valuation_params")
    if vp is not None:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Transaction Class", vp.known_class_name)
            c2.metric("Known PPS", f"${vp.known_share_price:,.4f}")
            c3.metric("Volatility", f"{vp.volatility:.0%}")
            c4, c5, c6 = st.columns(3)
            c4.metric("Risk-Free Rate", f"{vp.risk_free_rate:.2%}")
            c5.metric("Time to Liquidity", f"{vp.time_to_liquidity:.2f} yrs")
            c6.metric("DLOM", f"{vp.dlom_percent:.1%}")
    else:
        st.caption("No parameters saved yet — fill in the form above and click **Save Parameters**.")

# -------------------------------------------------------------------
# TAB 3: Results & Analysis
# -------------------------------------------------------------------
with tab_results:
    st.markdown('<h2 class="section-hdr">Results & Analysis</h2>', unsafe_allow_html=True)

    if st.session_state.get("run_requested"):
        st.session_state["run_requested"] = False
        ct = st.session_state.get("cap_table")
        vp = st.session_state.get("valuation_params")

        if ct is None:
            st.error("Please configure the **Cap Table** first (Tab 1).")
        elif vp is None:
            st.error("Please configure and save **Valuation Parameters** first (Tab 2).")
        else:
            with st.spinner("Running OPM backsolve…"):
                try:
                    result: ValuationResult = backsolve_equity_value(ct, vp)
                    st.session_state["valuation_result"] = result
                    st.success("Valuation complete!")
                except Exception as exc:
                    st.error(f"Valuation failed: {exc}")
                    st.session_state["valuation_result"] = None

    result: ValuationResult | None = st.session_state.get("valuation_result")
    vp = st.session_state.get("valuation_params")

    if result is None:
        st.info(
            "No results yet. Configure your cap table and valuation parameters, "
            "then click **🚀 Run Valuation** in the sidebar."
        )
    else:
        # 1. Key Metrics
        st.markdown("### 🔑 Key Metrics")
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f"""<div class="metric-card">
            <h3>Implied Total Equity Value</h3>
            <p>${result.total_equity_value:,.0f}</p>
            </div>""", unsafe_allow_html=True,
        )
        m2.markdown(
            f"""<div class="metric-card">
            <h3>Common Stock FMV (after DLOM)</h3>
            <p>${result.common_fmv:,.4f}</p>
            </div>""", unsafe_allow_html=True,
        )
        dlom_applied = vp.dlom_percent if vp else 0.0
        m3.markdown(
            f"""<div class="metric-card">
            <h3>DLOM Applied</h3>
            <p>{dlom_applied:.1%}</p>
            </div>""", unsafe_allow_html=True,
        )

        st.markdown("---")

        # 2. Per-Share Value Table
        st.markdown("### 📊 Per-Share Value Allocation")
        alloc_rows = []
        total_value = sum(result.class_total_values.values()) or 1.0
        ct = st.session_state.get("cap_table")
        for name, pps in result.per_share_values.items():
            tv = result.class_total_values.get(name, 0.0)
            # Find shares
            shares = 0
            for ec in ct.equity_classes:
                if ec.name == name:
                    shares = ec.shares_outstanding
            for op in ct.options:
                if op.name == name:
                    shares = op.shares_outstanding
            for w in ct.warrants:
                if w.name == name:
                    shares = w.shares_outstanding
            alloc_rows.append({
                "Class": name,
                "Shares": f"{shares:,.0f}",
                "Value Per Share (Pre-DLOM)": f"${pps:,.4f}",
                "Total Value": f"${tv:,.0f}",
                "% of Equity": f"{tv / total_value * 100:.1f}%",
            })
        st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 3. Waterfall Chart
        st.markdown("### 🏔️ Value Waterfall")
        st.info("The waterfall shows how total equity value flows through each breakpoint tranche.")
        bp_names = [f"BP {i}: ${bp:,.0f}" for i, bp in enumerate(result.breakpoints)]
        bp_values = result.tranche_values
        # Pad names for display
        waterfall_names = []
        waterfall_values = []
        for i, tv in enumerate(bp_values):
            if i < len(bp_names):
                waterfall_names.append(bp_names[i])
            else:
                waterfall_names.append(f"Above BP {len(bp_names)-1}")
            waterfall_values.append(tv)

        fig_waterfall = go.Figure(go.Waterfall(
            x=waterfall_names, y=waterfall_values,
            connector={"line": {"color": "#3b82f6"}},
            increasing={"marker": {"color": "#2563eb"}},
            decreasing={"marker": {"color": "#dc2626"}},
            totals={"marker": {"color": "#059669"}},
        ))
        fig_waterfall.update_layout(
            title="Breakpoint Waterfall", yaxis_title="Tranche Value ($)",
            xaxis_title="Breakpoint", showlegend=False,
            margin=dict(t=50, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # 4. Equity Value Allocation Pie
        st.markdown("### 🥧 Equity Value Allocation")
        pie_labels = [name for name, tv in result.class_total_values.items() if tv > 0]
        pie_values = [tv for tv in result.class_total_values.values() if tv > 0]
        if pie_labels:
            fig_alloc_pie = px.pie(
                names=pie_labels, values=pie_values,
                title="Total Value by Class",
                color_discrete_sequence=px.colors.sequential.Teal, hole=0.35,
            )
            fig_alloc_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_alloc_pie, use_container_width=True)

        st.markdown("---")

        # 5. Sensitivity Analysis
        st.markdown("### 🔥 Sensitivity Analysis")
        st.info("The heatmap shows how common-stock FMV varies with changes in volatility and time to liquidity.")

        base_vol = vp.volatility
        base_time = vp.time_to_liquidity

        vol_range = np.arange(max(0.10, base_vol - 0.20), base_vol + 0.21, 0.05)
        time_range = np.arange(max(0.5, base_time - 1.0), base_time + 2.01, 0.5)

        heat_data = np.zeros((len(vol_range), len(time_range)))
        for i, v in enumerate(vol_range):
            for j, t in enumerate(time_range):
                try:
                    sens_params = ValuationParams(
                        volatility=float(v),
                        risk_free_rate=vp.risk_free_rate,
                        time_to_liquidity=float(t),
                        known_share_price=vp.known_share_price,
                        known_class_name=vp.known_class_name,
                        dlom_percent=vp.dlom_percent,
                    )
                    sens_result = backsolve_equity_value(ct, sens_params)
                    heat_data[i, j] = sens_result.common_fmv
                except Exception:
                    heat_data[i, j] = np.nan

        fig_heat = go.Figure(go.Heatmap(
            z=heat_data,
            x=[f"{t:.1f}yr" for t in time_range],
            y=[f"{v:.0%}" for v in vol_range],
            colorscale="Blues",
            text=np.where(
                np.isnan(heat_data), "N/A",
                np.char.add("$", np.char.mod("%.4f", np.nan_to_num(heat_data))),
            ),
            texttemplate="%{text}",
            hovertemplate="Vol: %{y}<br>Time: %{x}<br>FMV: %{text}<extra></extra>",
            colorbar=dict(title="FMV ($)"),
        ))
        fig_heat.update_layout(
            title="Common Stock FMV Sensitivity",
            xaxis_title="Time to Liquidity", yaxis_title="Volatility",
            margin=dict(t=50, b=50, l=80, r=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # 6. Breakpoint Analysis Table
        st.markdown("### 🧮 Breakpoint Analysis")
        bp_rows = []
        for i in range(len(result.allocations)):
            bp_val = result.breakpoints[i] if i < len(result.breakpoints) else "∞"
            bp_next = result.breakpoints[i + 1] if i + 1 < len(result.breakpoints) else "∞"
            tv = result.tranche_values[i] if i < len(result.tranche_values) else 0.0
            alloc = result.allocations[i]
            alloc_detail = ", ".join(f"{k}: {v:.1%}" for k, v in alloc.items() if v > 0)
            bp_rows.append({
                "Tranche": i + 1,
                "From ($)": f"${bp_val:,.0f}" if isinstance(bp_val, (int, float)) else bp_val,
                "To ($)": f"${bp_next:,.0f}" if isinstance(bp_next, (int, float)) else bp_next,
                "Tranche Value ($)": f"${tv:,.2f}",
                "Allocation": alloc_detail or "—",
            })
        st.dataframe(pd.DataFrame(bp_rows), use_container_width=True, hide_index=True)
