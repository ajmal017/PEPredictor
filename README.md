PEPredictor
===========

Parses through a data file containing attributes (Return, Volatility, Relative Strength etc) for US Equity Tickers. Uses the KNN algorithm to find the nearest neighbor for a given stock and determines its Predicted fundamental ratios (P/E, P/B and P/S). Also generates a 95% price confidence range using yearly volatility for the equity.

To Do:

1) Plot a pie chart for the influencing neighbors with their relevant percentage influences.

2) Implement a webservice call to retrieve real time CSV files containing the data.