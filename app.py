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

# FIX: Prevent yfinance from crashing on Streamlit Cloud cache locks
try:
    yf.set_tz_cache_location("/tmp/yfinance_tz_cache")
except Exception:
    pass

# ==============================================================================
# PAGE SETUP & CONFIG
# ==============================================================================
st.set_page_config(page_title="Portfolio Risk & Hedge AI", layout="wide", page_icon="📈")
load_dotenv()

# ==============================================================================
# CORE MATH & DATA FUNCTIONS (Cached for web speed)
# ==============================================================================
@st.cache_data(ttl=3600) 
def download_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        return pd.DataFrame()
        
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close']
        else:
            prices = data['Close']
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
    price_paths =
