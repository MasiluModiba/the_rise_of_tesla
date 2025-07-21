
# 📈 Tesla Stock Price Analysis and Forecasting using Machine Learning and Time Series Techniques

This repository contains a comprehensive, research-driven analysis of Tesla Inc.’s (TSLA) historical stock performance, focusing on exploratory data analysis (EDA), feature engineering, and predictive modeling. Using over nine years of stock price data, we build and evaluate traditional statistical tools alongside advanced machine learning models to understand, predict, and simulate investment strategies based on Tesla’s price behavior.

---

## 🗂 Project Overview

### 📌 Motivation
Tesla (TSLA) is a high-profile, volatile, and narrative-driven stock frequently affected by social sentiment, tweets, media narratives, and broader tech trends. Its explosive growth and sharp corrections make it ideal for studying the limitations and opportunities in stock forecasting. This project combines financial domain knowledge, statistical analysis, and machine learning to simulate real-world investment strategy development.

### 🎯 Objectives
- Understand the dynamics of Tesla’s price history: identifying key trends, volatility regimes, and turning points.
- Engineer features and build forecasting models to predict Tesla’s next-day closing price.
- Compare traditional methods with machine learning techniques (e.g., XGBoost, Linear Regression).
- Benchmark Tesla’s performance against indices: Nasdaq and S&P 500.
- Provide data-driven insights into portfolio allocation and risk-adjusted investing.

### 📊 Dataset Overview
- **Source:** Kaggle (Tesla Daily Historical Data)
- **Period Covered:** January 1, 2015 – January 16, 2024
- **Fields:** Date, Open, High, Low, Close, Volume
- **Cleanliness:** No missing values in critical fields; data was formatted, indexed, and enriched with technical indicators.

---

## 🔍 Exploratory Data Analysis (EDA)

### Volatility Regimes & Rolling Analysis
- Tesla’s price history is divided into major phases:
  - **2015–2019:** Stable with modest gains.
  - **2020–2021:** Parabolic growth fueled by speculative buying.
  - **2022–2024:** Correction phase with large drawdowns and volatility spikes.
- These phases reflect shifts in sentiment, liquidity, and macroeconomic policy.
![alt text](image.png)

### Price Behavior & Statistical Summary
- **Mean Log Return:** Positive, suggesting an upward drift in long-term price behavior.
- **Volatility:** ~3.5% daily, much higher than the S&P 500 and Nasdaq, confirming the high-risk profile.
- **Skewness & Kurtosis:** Slightly negative skew and fat-tailed distribution; extreme movements are more likely than normal.
- **Normality Tests:** Shapiro-Wilk reject normality, validating the need for non-parametric modeling.

![alt text](image-1.png)

### Technical Indicators 
- Moving Averages (MA20, MA100), Bollinger Bands, and VWAP added to understand trend-following and mean-reversion behaviors.
- Indicators helped flag overbought/oversold conditions and periods of breakout or consolidation.
- Granger Causality Tests show that volume has no significant effect on closing prices
- Autocorrelation and Lag plot tests, show a strong correlation between Tesla's closing ptices and past values
- Statistical summary of closing prices indicate Tesla's log returns are not normally distributed, show high volatility and a long term upward drift.
- Distribution of closing prices show right skewness indicating rapid growth over time and low frequency of high prices show that the stock reached it's existing highs receltly after prolonged growth.


### Comparative Benchmarking
- Tesla returned **~31% annually**, compared to **13% for Nasdaq** and **9% for S&P 500**.
- Maximum drawdown: Tesla = -73.6%; S&P 500 = -34%.
- Despite extreme volatility, Tesla's **Sharpe ratio = 0.86**, higher than both indices.
- At the same time 30-Day rolling volatility Tesla showed significantly higher volatility throughout, highlighting its status as a high risk high reward equity
- Correlation tests show Tesla has weaker correlation with other tech giants such as Apple and Meta and even Indices suggesting that it behaves differently than the broader tech sector.
- Analysis of fundementals shows Tesla's valuation is exceptionally high with a P/E ratio of ~188, greater than other high growth equities like Nvidia (~56), however it has a modest EPS relative to maga-cap peers like Microsoft and Meta.
It's beta is nearly double the market's volatility showing extreme sensitivity to market swings

### Portfolio Simulation
Portfolio Performance Metrics (2015–2024):

Equal Weight:
  Annual Return:    24.02%
  Annual Volatility:27.98%
  Sharpe Ratio:     0.86

TSLA Heavy:
  Annual Return:    32.77%
  Annual Volatility:38.52%
  Sharpe Ratio:     0.85

Index Heavy:
  Annual Return:    19.65%
  Annual Volatility:23.69%
  Sharpe Ratio:     0.83

- An equal-weighted portfolio of Tesla, Nasdaq, and S&P 500 delivered strong long-term returns but with higher volatility than the Index heavy portfolio. Tesla heavy portfolio showed the highest growth potential but also increased risk and drawdowns. For most investors, combining Tesla with broad market indices offers a balanced approach which are higher returns with manageable risk, provided allocations match risk tolerance. Sharpe ratio remained relatively the same across the portfolios.

---

## Feature Engineering & Predictive Modeling

### Features Used
- **Lagged prices**: Previous day Close, High, Low, Open
- **Returns**: Daily and rolling log returns
- **Volatility**: Rolling standard deviations
- **Technical indicators**: Bollinger bands, VWAP, moving averages
- **Calendar effects**: Day of week, month, etc.

### Model Selection & Evaluation
Three models were developed:
- **Linear Regression**: Simple, interpretable baseline
- **Random Forest Regressor**: Nonlinear, ensemble-based approach with better accuracy
- **XGBoost Regressor**: Gradient-boosted model yielding the best results

| Model            | RMSE     | R² Score |
|------------------|----------|----------|
| Linear Regression| 13.52    | 0.56     |
| Random Forest    | 9.21     | 0.74     |
| **XGBoost**      | **7.97** | **0.85** |

- XGBoost captured short-term dependencies and nonlinear interactions effectively.
- Feature importance plots show heavy reliance on intraday features (High, Low, Open) and recent lags.

---

## Advanced XGBoost Tuning

### Model Optimization
Hyperparameters were tuned to maximize out-of-sample accuracy. Cross-validation was used with time-series split to avoid leakage. XGBoost maintained strong generalization on the test set and avoided overfitting.

### Feature Importance
Key drivers included:
- 1-day lag Close
- Intraday range (High – Low)
- Previous day's Open and Volume
- Technical levels (VWAP, MA100)

However, models struggled to incorporate broader market trends or macroeconomic signals, emphasizing the limits of price-based features alone.

---

## In-depth Results & Portfolio Recommendations

### Financial Insights
- Tesla’s price is highly autocorrelated, meaning today’s price is strongly linked to yesterday’s.
- Returns are not normal and not white noise, which supports model-driven forecasting.
- Drawdowns are deep, requiring investor discipline and hedging strategies.

### Portfolio Strategy
- Risk-tolerant investors may benefit from holding TSLA as a **satellite asset**.
- Conservative investors should limit exposure and diversify using low-beta assets like Apple or index ETFs.
- Use technical models and macro filters (interest rate regime, Fed policy, etc.) for entry/exit timing.

---

## Challenges & Limitations

### Structural Challenges
- **Non-stationarity**: Tesla's regime shifts make long-term predictions unreliable.
- **Speculative Behavior**: Sentiment and news often drive price more than fundamentals.
- **Oscillatory and Collapsing Forecasts**: Many models failed to extrapolate trends, reverting to unrealistic baselines.

### Modeling Issues
- **Noise Overfitting**: Models overfit on recent noise, failing to generalize.
- **Flat Forecasts**: Prophet and others defaulted to flat lines post-training.
- **Real-World Utility**: Models with low RMSE sometimes gave useless financial predictions (e.g., flat prices).

### Practical Considerations
- Need for **exogenous features**: Macroeconomic indicators, social media sentiment, or earnings data could enhance model robustness.

---

## Conclusion

Tesla’s price dynamics present a unique opportunity and challenge. While advanced machine learning models like XGBoost offer meaningful improvements in prediction accuracy, they fall short when faced with structural changes and non-price drivers.

- TSLA is **not suitable for passive investment**.
- A **quantitative and discretionary approach** is needed.
- For analysts, Tesla is a **laboratory for testing model limits** and adaptive strategies.

The project underscores the importance of blending **financial knowledge** with **data science best practices** to achieve robust, interpretable, and actionable insights.

---

## 📁 Files in this Repository

- `Tesla Stock Data.ipynb`: Full notebook with analysis, visualizations, and models
- `README.md`: This summary and extended documentation
- `requirements.txt`