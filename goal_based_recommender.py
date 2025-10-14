# goal_based_recommender.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
import cvxpy as cp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Goal-Based Stock Recommender", layout="wide")

st.title("🎯 Goal-Based Stock Recommender")
st.markdown("""
This app recommends a **goal-oriented stock portfolio** based on your target amount, investment horizon, 
and risk preference — using **Modern Portfolio Theory** and **Monte Carlo simulation**.
""")

# --- Sidebar: User inputs ---
st.sidebar.header("Your Investment Goal")
current_capital = st.sidebar.number_input("Current Investment ($)", value=10000.0, step=1000.0)
monthly_contrib = st.sidebar.number_input("Monthly Contribution ($)", value=200.0, step=50.0)
target_amount = st.sidebar.number_input("Target Amount ($)", value=20000.0, step=1000.0)
years = st.sidebar.slider("Time Horizon (Years)", 1, 30, 5)
risk_tolerance = st.sidebar.select_slider("Risk Tolerance", options=["Low", "Medium", "High"], value="Medium")
simulate_button = st.sidebar.button("🚀 Run Recommendation")

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit & Python")

# --- Asset Universe ---
default_tickers = ["SPY", "VTI", "IWM", "QQQ", "TLT", "AGG"]
tickers = st.multiselect("Choose investment universe (ETFs/stocks):", default_tickers, default=default_tickers)

if simulate_button:
    try:
        with st.spinner("Fetching market data and running analysis..."):
            # --- 1) Download price data ---
            prices = yf.download(tickers, start="2015-01-01", end=None)["Adj Close"].dropna()
            returns = prices.resample("M").last().pct_change().dropna()

            # --- 2) Estimate expected returns and covariance ---
            mu = returns.mean()
            lw = LedoitWolf().fit(returns.values)
            cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

            # Adjust expected return based on risk tolerance
            risk_multipliers = {"Low": 0.7, "Medium": 1.0, "High": 1.3}
            mu = mu * risk_multipliers[risk_tolerance]

            # --- 3) Markowitz Optimization ---
            n = len(tickers)
            w = cp.Variable(n)
            Sigma = cov.values
            mu_vec = mu.values
            target_monthly_return = np.mean(mu_vec) * risk_multipliers[risk_tolerance]

            prob = cp.Problem(
                cp.Minimize(cp.quad_form(w, Sigma)),
                [
                    cp.sum(w) == 1,
                    w >= 0,
                    w @ mu_vec >= target_monthly_return
                ]
            )
            prob.solve()
            weights = np.array(w.value).flatten()

            # --- 4) Monte Carlo Simulation ---
            steps = int(years * 12)
            sim_paths = 5000
            mean = mu_vec
            cov_mat = Sigma
            chol = np.linalg.cholesky(cov_mat)
            final_vals = np.zeros(sim_paths)

            for p in range(sim_paths):
                wealth = current_capital
                for t in range(steps):
                    z = np.random.normal(size=n)
                    r = mean + chol @ z
                    portfolio_r = np.dot(weights, r)
                    wealth = wealth * (1 + portfolio_r) + monthly_contrib
                final_vals[p] = wealth

            prob_success = np.mean(final_vals >= target_amount)
            median_val = np.median(final_vals)
            p10 = np.percentile(final_vals, 10)
            p90 = np.percentile(final_vals, 90)

            # --- 5) Display Results ---
            st.subheader("📊 Recommended Portfolio")
            results_df = pd.DataFrame({
                "Ticker": tickers,
                "Weight": np.round(weights, 3)
            })
            st.dataframe(results_df, use_container_width=True)

            st.subheader("🎯 Goal Projection Summary")
            st.markdown(f"""
            - **Target Amount:** ${target_amount:,.0f}  
            - **Time Horizon:** {years} years  
            - **Probability of Success:** {prob_success:.1%}  
            - **Median Terminal Wealth:** ${median_val:,.0f}  
            - **10th–90th Percentile Range:** ${p10:,.0f} – ${p90:,.0f}
            """)

            # --- 6) Chart ---
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(final_vals, bins=40, alpha=0.7)
            ax.axvline(target_amount, color="red", linestyle="--", label="Target")
            ax.set_xlabel("Final Portfolio Value ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

            st.info("📘 Disclaimer: This is a research and educational demo. "
                    "Past performance is not indicative of future results. "
                    "This is not personalized financial advice.")

    except Exception as e:
        st.error(f"An error occurred: {e}")



