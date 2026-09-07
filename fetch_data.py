import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def fetch_prices(tickers, period="3y"):
    """Pulls daily closing prices for a list of tickers."""
    data = yf.download(tickers, period=period)["Close"]
    return data

def compute_daily_returns(prices):
    """Converts daily prices into daily % returns."""
    returns = prices.pct_change().dropna()
    return returns

# Test tickers
tickers = ["NVDA", "AMZN", "KO", "JPM"]
prices = fetch_prices(tickers)
returns = compute_daily_returns(prices)

mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("=== OPTIMAL ALLOCATION (%) ===")
for ticker, weight in cleaned_weights.items():
    print(f"{ticker}: {weight * 100:.1f}%")

expected_return, volatility, sharpe = ef.portfolio_performance(verbose=True)

# --- NEW: convert percentages into actual dollars/shares ---
investment_amount = 10000  # this will become user input later

latest_prices = get_latest_prices(prices)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=investment_amount)
allocation, leftover_cash = da.greedy_portfolio()

print(f"\n=== ALLOCATION FOR ${investment_amount} ===")
for ticker, shares in allocation.items():
    dollar_amount = shares * latest_prices[ticker]
    print(f"{ticker}: {shares} shares (~${dollar_amount:.2f})")

print(f"\nLeftover cash (couldn't be invested in whole shares): ${leftover_cash:.2f}")