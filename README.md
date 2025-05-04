# Capstone Project Proposal: Investment Trading with Yahoo Finance Data
### 1. Domain Background
Investment trading is a critical domain in finance, involving decision-making based on economic
indicators, market sentiment, and statistical signals. Traditionally dominated by manual and
heuristic methods, modern trading increasingly relies on machine learning (ML) models for
predictive insights and automated decision-making. With the availability of historical financial
data from platforms such as Yahoo Finance, there is a unique opportunity to apply ML
algorithms to forecast stock price movements and improve trading strategies.
### 2. Problem Statement
The financial market is inherently volatile and non-linear, making it difficult to predict short-term
price movements. The core problem to be addressed in this project is: Can we predict the
next-day direction (up or down) of a stock's closing price using technical indicators derived from
historical data? This is a binary classification task where the performance of the model can
significantly impact trading outcomes and profitability.
### 3. Solution Statement
This project proposes the development of a machine learning classification model to predict
next-day stock price direction. The solution will leverage engineered features from historical
OHLCV (Open, High, Low, Close, Volume) data, including technical indicators such as MACD,
RSI, SMA, and Bollinger Bands. Models such as Random Forest, XGBoost, and LSTM will be
explored to capture both non-linear relationships and temporal dependencies. The final solution
will be evaluated not only on classification metrics but also on its effectiveness in simulated
trading (e.g., ROI, Sharpe Ratio).

## Installations
The following libraries are needed to successfully run this project. First of all you will need to install `yfinance` using `pip install yfinance`.
I used python 3 for this project and other packages installed are:

1. sklearn
2. seaborn
3. matplotlib
4. pandas
5. numpy
6. tensorflow