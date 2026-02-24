
import os
import json
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE SETUP & CONFIG
# ==============================================================================
st.set_page_config(page_title="Portfolio Risk & Hedge AI", layout="wide", page_icon="📈")
load_dotenv()

# ==============================================================================
# CORE MATH & DATA FUNCTIONS (Cached for web speed)
# ==============================================================================
@st.cache_data(ttl=3600) # Cache the data for 1 hour so the web app is fast
def download_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    return prices.dropna()

def compute_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def rolling_volatility(returns, window=21):
    return returns.rolling(window=window).std() * np.sqrt(252)

def correlation_matrix(returns):
    return returns.corr()

def beta_exposure(returns, benchmark_col="SPY"):
    betas = {}
    X = returns[benchmark_col].values.reshape(-1, 1)
    lr = LinearRegression()
    for col in returns.columns:
        if col != benchmark_col:
            y = returns[col].values
            lr.fit(X, y)
            betas[col] = float(lr.coef_[0])
    return betas

def historical_var(returns, weights, confidence_level=0.95):
    assets = list(weights.keys())
    w_array = np.array([weights[ticker] for ticker in assets])
    port_returns = returns[assets].dot(w_array)
    return float(np.percentile(port_returns, 100 * (1 - confidence_level)))

def monte_carlo_drawdown(returns, weights, n_simulations=1000, n_days=252):
    assets = list(weights.keys())
    w_array = np.array([weights[ticker] for ticker in assets])
    port_returns = returns[assets].dot(w_array)
    mu = port_returns.mean()
    sigma = port_returns.std()
    
    simulated_returns = np.random.normal(mu, sigma, (n_days, n_simulations))
    price_paths = np.exp(np.cumsum(simulated_returns, axis=0))
    rolling_max = np.maximum.accumulate(price_paths, axis=0)
    drawdowns = (price_paths - rolling_max) / rolling_max
    return drawdowns.min(axis=0)

# ==============================================================================
# GEMINI AI INTEGRATION
# ==============================================================================
def generate_hedge_plan(features_dict, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are a quantitative portfolio manager and risk specialist.
    Analyze the provided portfolio risk metrics. 
    Propose a targeted hedging strategy using standard US Sector ETFs, Gold (GLD), or Silver (SLV).
    
    You MUST output your response STRICTLY as a valid JSON object matching this schema exactly, with no markdown:
    {{
        "risk_assessment": "Detailed summary of current portfolio risks based on the data.",
        "concentration_warning": "Identification of any dangerous sector, beta, or correlation concentrations.",
        "proposed_hedges": [
            {{
                "instrument": "Ticker symbol of recommended hedge (e.g. XLU, GLD, SLV)", 
                "weight_adjustment_pct": "Float representing suggested portfolio allocation % to this hedge", 
                "reason": "Specific quantitative justification for this hedge based on the provided correlations and betas."
            }}
        ]
    }}
    
    Portfolio Risk Metrics:
    {json.dumps(features_dict, indent=2)}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", 
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

# ==============================================================================
# STREAMLIT UI LAYOUT
# ==============================================================================
st.title("🛡️ Portfolio Risk & Hedge AI ")
st.markdown("Analyzing end-of-day market conditions and generating AI-driven hedge optimizations.")

# --- SIDEBAR: User Inputs ---
with st.sidebar:
    st.header("Portfolio Parameters")
    
    # API Key Input (Fallback to .env)
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", value=env_key, type="password")
    
    st.subheader("Asset Allocations (%)")
    w_aapl = st.slider("AAPL (Apple)", 0.0, 1.0, 0.25)
    w_msft = st.slider("MSFT (Microsoft)", 0.0, 1.0, 0.25)
    w_xle = st.slider("XLE (Energy ETF)", 0.0, 1.0, 0.10)
    w_xlk = st.slider("XLK (Tech ETF)", 0.0, 1.0, 0.20)
    w_gld = st.slider("GLD (Gold)", 0.0, 1.0, 0.10)
    w_slv = st.slider("SLV (Silver)", 0.0, 1.0, 0.10)
    
    weights = {
        "AAPL": w_aapl, "MSFT": w_msft, "XLE": w_xle, 
        "XLK": w_xlk, "GLD": w_gld, "SLV": w_slv
    }
    
    # Normalize weights if they don't equal 100%
    total_w = sum(weights.values())
    if total_w != 1.0 and total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
        st.warning(f"Weights normalized to 100%. (Sum was {total_w:.2f})")

# --- MAIN DASHBOARD ---
tickers = list(weights.keys()) + ["SPY"]

with st.spinner("Fetching steady-state market data..."):
    # We use yfinance here instead of Alpha Vantage so the web dashboard loads instantly
    prices = download_prices(tickers, start_date="2023-01-01", end_date="2024-01-01")
    returns = compute_returns(prices)

# Calculate Metrics
roll_vol = rolling_volatility(returns)
corr_mat = correlation_matrix(returns)
betas = beta_exposure(returns, benchmark_col="SPY")
var_95 = historical_var(returns, weights, confidence_level=0.95)
drawdowns = monte_carlo_drawdown(returns, weights, n_simulations=2000)

# 1. Top Level Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Historical 95% VaR (Daily)", f"{var_95:.2%}", "Risk of Loss", delta_color="inverse")
col2.metric("Worst-Case Simulated Drawdown (5%)", f"{np.percentile(drawdowns, 5):.2%}", "Stress Test", delta_color="inverse")
col3.metric("Highest Beta Asset", f"{max(betas, key=betas.get)} ({max(betas.values()):.2f})")

st.markdown("---")

# 2. Interactive Charts (Plotly)
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Asset Correlation Matrix")
    fig_corr = px.imshow(corr_mat.loc[list(weights.keys()), list(weights.keys())], 
                         text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

with col_chart2:
    st.subheader("Monte Carlo Max Drawdown (1-Year)")
    fig_mc = go.Figure(data=[go.Histogram(x=drawdowns, nbinsx=50, marker_color='#E74C3C')])
    fig_mc.update_layout(xaxis_title="Max Drawdown", yaxis_title="Frequency", showlegend=False)
    st.plotly_chart(fig_mc, use_container_width=True)

st.subheader("21-Day Rolling Volatility (Annualized)")
fig_vol = px.line(roll_vol[list(weights.keys())].dropna())
fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volatility", legend_title="Assets")
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")()

# ==============================================================================
# AI HEDGE GENERATOR TRIGGER
# ==============================================================================
st.header("🧠 Gemini AI Hedge Optimizer")

if st.button("Generate Hedge Strategy based on Steady-State Data", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar to run the AI.")
    else:
        with st.spinner("Analyzing portfolio macro-structure with Gemini AI..."):
            features = {
                "portfolio_weights": weights,
                "market_betas": betas,
                "daily_historical_var_95": var_95,
                "simulated_worst_case_drawdown_5pct": float(np.percentile(drawdowns, 5)),
                "asset_correlations": corr_mat.round(3).to_dict()
            }
            
            try:
                plan = generate_hedge_plan(features, api_key)
                
                # Display parsed JSON beautifully in the UI
                st.info(f"**Risk Assessment:** {plan.get('risk_assessment', 'N/A')}")
                st.warning(f"**Concentration Warning:** {plan.get('concentration_warning', 'N/A')}")
                
                st.subheader("Proposed Executions")
                for hedge in plan.get("proposed_hedges", []):
                    with st.expander(f"Buy/Adjust: **{hedge.get('instrument')}** (Target: {hedge.get('weight_adjustment_pct')}%)", expanded=True):
                        st.write(f"**Quantitative Rationale:** {hedge.get('reason')}")
                        
            except Exception as e:
                st.error(f"Failed to generate AI plan. Error: {e}")
